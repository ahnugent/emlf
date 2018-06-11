# Script:       mlfns.R
# Authors:      Allen H. Nugent, 2017+
# Last edit:    2018-06-06
# Last test:    2018-06-06
#
# Purpose:      Library of machine learning utilities.
#
# Notes:
#
#    1. The typical ML experimental sequence is as follows:
#
#           Step 1: Prepare data, decide on predictors and response variable ('labels'), 
#                   algorithm, and data subsetting.
#
#           Step 2: Call setupModel() to instantiate the `model` object and populate it 
#                   in preparation for training. 
#
#                a) If specified, member data structures will be created in preparation 
#                   for cross-validation. 
#
#                b) Supported transformations can be applied to all predictor variables.  
#                   Unsupported (custom) transformations or selective application of  
#                   transformations (to specific columns) must be coded manually, in the 
#                   next step.
#
#           Step 3: Apply custom transformations to `model$mdat` columns as required.
#
#           Step 4: Invoke the model training function associated with the model.
#
#                   If cross-validation is being performed, training should be executed 
#                   in a loop that assigns the the `cv` members.
#
#           Step 5: Call getModelPredictions() to invoke the default predict() function 
#                   on the test data. (The predictions vector is appended to the `model` 
#                   object.)
#
#           Step 6: For classification problems, call getModelClasses() to convert the  
#                   predicted probabilities to classes, with optional threshold setting. 
#                   (The vector of predicted classes is appended to the `model` object).
#
#           Step 7: Call evaluateModel() to return (and, optionally, print) the results 
#                   of a suite of model evaluation functions (from various CRAN packages).
#
#    2. The setupModel() function copies model the data frame `model$mdat` to 
#       `model$mdat.raw` prior to applying any transformations.
#
#    3. The `model` object can be extended by custom members with no risk of unintended 
#       consequence, so long as no naming clashes occur.
#
# Contents:
#
#   Accuracy				Returns simple classification accuracy.
#
#   accuracy.rating         Returns a string that summarises the effectiveness of a predictor 
#                           based on the provided accuracy metric.
#
#   assignClass             Assigns a vector of class memberships / levels based on a threshold.
#
#   cluster.summary         Summarises the results of fitting a clustering model.
#
#   drop_na                 Removes data rows with NAs from model (including corresponding label rows).
#
#   evaluateModel           Subjects a model to a series of tests, partitioning into training 
#                           and testing subsets (if specified).
#
#   getModelClasses			Assigns specified classes using predicted probabilities.
#   
#   getModelPredictions		Invokes the predict() function and returns the vector of predictions.
#
#   getSimpleClasses        Returns a vector of (simple) column classes.
#
#   Kappa					Returns the 'kappa' classification accuracy metric.
#
#   misClassError           Simple misclassification rate.
#
#   normalise               Rescales a vector according to the range of the positive side, negative side, 
#                           the greater of the two, or the sum of the two.
#
#   plotPredictorStats      Plots a statistic of predictor performance (relative significance or power).
#
#   setupModel              Instantiates the model object -- a list that contains descriptive information,
#                           training data, and parameters.
#
#   trainModel              Invokes a specified ML algorith and appends the output object to the model 
#                           object.
#
#   vector.indices          Returns indices of a named vector or data frame that match the input names.
#
#
# TODO: create trainModel() fn (?!)
#           params = training alo name (and package name), algo params
# TODO: incorporate CV to evaluateModel()
# TODO: overloaded S3 print method


require(EMT)
require(ModelMetrics)
require(pscl)
require(Hmisc)
require(e1071)

# pseudoconstants:
tag_col.external_ <- '(external)'
tag_col.row_numbers_ <- '(row numbers)'


assignClass <- function(x, classes = 'boolean', threshold = 0.5) {
    
    # Assigns a vector of classes / levels based on a threshold probability.
    
    # classes       type of response required
    #                   'boolean' returns 0 or 1
    #                   'character' returns 'FALSE' or 'TRUE'
    #                   'logical' returns FALSE or TRUE
    
    x <- as.numeric(x)   # strip names
    
    if (classes %in% c('boolean', 'character', 'logical')) {
        n.classes <- 2
    } else {
        n.classes <- NROW(threshold)
    }
    
    # TODO: support multiple thresholds or other schemes for multinomial categorisation

    if (n.classes == 2) {
        
        # this only works for bivariate classes: 
        f <- function(a, threshold) {
            return(a >= threshold)
        }
        
        # FALSE/TRUE:
        out <- sapply(x, f, threshold) 
        
        # 'FALSE'/'TRUE':
        if (classes == 'character') {
            out <- ifelse(out, 'TRUE', 'FALSE')
        }
        # 0/1:
        if (classes == 'boolean') {
            out <- ifelse(out, 1, 0)
        }
        
        return(out)        
    }
}


getModelPredictions <- function(inmodel, pdat = NULL) {
    
    # Returns model predictions as a vector or an augmented data frame.
    
    # inmodel is either a model object 
    #   (and pdat is a logical vector of row selections, or NULL for all rows) 
    # or (deprecated) the output of a caret training function 
    #   (and pdat is the data.frame of predictors).
    #
    # If inmodel$class == 'encmodel':
    #   Invokes the predict() function on inmodel$mdat[pdat, ]
    #   and assigns the output to inmodel$yhat
	# Else:
    #   Invokes the predict() function for data.frame pdat 
    #   and returns the vector of predictions.
	
    return.what <- function (inmodel, yhat) {
        if (!is.null(inmodel$class)) {  #  if (inmodel$class == 'encmodel')
            inmodel$yhat <- yhat
            return(inmodel)
        } else {
            return(yhat)
        }
    }
    
    if (!is.null(inmodel$class)) {  #  if (inmodel$class == 'encmodel')
        fit <- inmodel$fit
        if (is.null(pdat)) {
            dat <- inmodel$mdat[, inmodel$predictor_cols]
        } else {
            dat <- inmodel$mdat[pdat, inmodel$predictor_cols]
        }
    } else {
        fit <- inmodel
        dat <- pdat
    }
    
    # TODO: What special cases / parameters are required by the various incarnations of predict()?
    
    if (class(fit)[1] == c('glm')) {
        yhat <- predict(fit, dat, type = 'response')
        return(return.what(inmodel, yhat))
    } 
    
    if (class(fit)[1] == c('naiveBayes')) {
        yhat <- as.numeric(predict(fit, dat, type = 'raw'))
        return(return.what(inmodel, yhat))
    } 
}


getModelClasses <- function(inmodel, response_type = NULL, threshold = 0.5) {
    
	# Assigns specified classes using predicted probabilities.
    
    # inmodel is either a model object or a vector of probabilities.
    #
    # if model$class == 'encmodel':
    #   assigns classes for inmodel$yhat, appends results as inmodel$yhat.class
    # else:
    #   assigns classes for inmodel, returns results as a vector.
    #
    # threshold should be set after examining ROC curve.
    
    # if (!is.null(inmodel$class)) {  #  if (inmodel$class == 'encmodel')
    
    inmodel_is_vector <- (class(inmodel) == 'numeric')
    
    if (inmodel_is_vector) {
        yhat <- inmodel
        model.response_type <- response_type
    } else {
        yhat <- inmodel$yhat
        model.response_type <- inmodel$response_type
    } 
    
    # TODO: transformation required depending on model$response_type = 'logical', 'categorical', or 'continuous' ?
    
    if (model.response_type == 'logical') {
        yhat.class <- assignClass(yhat, 'logical')
    }
    
    if (model.response_type == 'boolean') {
        yhat.class <- assignClass(yhat, 'boolean')
    }
    
    if (inmodel_is_vector) {  
        return(yhat.class)
    } else {
        inmodel$yhat.class <- yhat.class
        return(inmodel)
    }
}


evaluateModel <- function(model, partitions, verbose = FALSE,
                          evals = c('anova', 'auc', 'confusionMatrix', 'misClassError',
                                    'pR2', 'summary', 'varImp')) 
{
    
    # Subjects a model to a series of tests and appends output to model object.
    #
    # Parameters:
    #
    #   partitions  a character string vector containing any/all of c('training', 'validation', 'testing')
    #   evals       a character string vector containing any combination of supported test names.
    #
    # Tests   
    #
    #   Accuracy            = defined locally
    #   anova               = stats::anova
    #   auc                 = ModelMetrics::auc
    #   confusionMatrix     = caret::confusionMatrix
    #   Kappa               = defined locally
    #   misClassError       = defined locally
    #   varImp              = caret::varImp
    #
    # TODO: add BIC (make AIC & BIC default selections for cross-validation models)
    # TODO: determine which tests work with which algos: wrap in logic 
    #
    
    # WARNING: sensitivity & specificity in caret::confusionMatrix get swapped by default; must set positive = 'TRUE' 
    
    out <- list()
    out$training <- list()
    out$testing <- list()
    
    get.model.accuracy <- function(model, partition) {
        result <- list()
        if (class(partition) == 'numeric') {
            eval.rows <- partition
        } else {
            if (partition == 'training') {
                eval.rows <- model$training
            }
            if (partition == 'testing') {
                eval.rows <- model$testing
            }
            if (partition == 'validation') {
                eval.rows <- model$validation
            }
        }
        
        if (is.null(model$yhat.class)) {
            cat('evaluateModel(): No class predictions found in model.\n')
        } else {
            mdat <- model$mdat[eval.rows, ]
            mlabs <- as.factor(model$mlabs[eval.rows])
            yhat <- model$yhat[eval.rows]
            yhat.class <- model$yhat.class[eval.rows]

            if (sum(is.na(yhat.class) > 0)) {
                warning("evaluateModel(): NA's in predicted classes.")
                rows.keep <- !is.na(yhat.class)
                mdat <- mdat[rows.keep, ]
                mlabs <- mlabs[rows.keep]
                yhat <- yhat[rows.keep]
                yhat.class <- yhat.class[rows.keep]
            }
            
            if (class(yhat.class) == 'logical') {
                yhat.fclass <- as.factor(yhat.class)
                fmlabs <- as.factor(mlabs)
            } else {
                yhat.fclass <- as.factor(bool.to.logical(yhat.class))
                fmlabs <- as.factor(bool.to.logical(mlabs))
            }

            if ('misClassError' %in% evals) {
                result$misClassError <- misClassError(yhat.class, mlabs)
            }
            
            if ('confusionMatrix' %in% evals) {
                result$confusionMatrix <- caret::confusionMatrix(data = yhat.fclass, 
                                                             reference = fmlabs, positive = 'TRUE', 
                                                             dnn = c("Prediction", "Reference"))
            }
            if ('auc' %in% evals) {
                result$auc <- ModelMetrics::auc(as.factor(mlabs), yhat)
            }
            
            if ('varImp' %in% evals) {
                if (model$algo_code %in% c('glm')) {
                    result$varImp <- varImp(model$fit)
                } else {
                    if (verbose) {
                        cat('evaluateModel(): varImp() not supported for model class', class(model$fit), '\n')
                    }
                }
            }
        }
        
        return(result) 
    }
    
    # model performance:
    
    if ('summary' %in% evals) {
        out$summary <- summary(model$fit)
    }
    
    if ('anova' %in% evals) {
        if (class(model$fit)[1] == 'glm') {
            out$anova <- anova(model$fit, test = "Chisq")
        } else {
            if (verbose) {
                cat('evaluateModel(): anova() not supported for model class', class(model$fit), '\n')
            }
        }
    }
    
    if ('pR2' %in% evals) {
        if (class(model$fit)[1] == 'glm') {
            out$pR2 <- pR2(model$fit)
        } else {
            if (verbose) {
                cat('evaluateModel(): pR2() not supported for model class', class(model$fit), '\n')
            }
        }
    }
    
    if (verbose) {
        print(out$summary)
        print(out$anova)
        print(out$pR2)
    }
    
    if (class(partitions) == 'numeric') # partitions is a vector of indices (for a single fold of data):
    {   
        out$fold <- get.model.accuracy(model, partitions)  
    } else {
        if ('training' %in% partitions) {
            out$training <- get.model.accuracy(model, 'training') 
            if (verbose) {
                cat('Training accuracy =', get.annotation(1 - out$training$misClassError, 
                                                          what = 'percent', parentheses = FALSE), '\n')
                cat('Training confusion matrix: \n \n')
                print(out$training$confusionMatrix)
                cat('Training AUC: \n')
                print(out$training$auc)
                cat('\n')
            }
        }
        if ('testing' %in% partitions) {
            out$testing <- get.model.accuracy(model, 'testing') 
            if (verbose) {
                cat('Testing accuracy =', get.annotation(1 - out$testing$misClassError, 
                                                         what = 'percent', parentheses = FALSE), '\n')
                cat('Testing confusion matrix: \n \n')
                print(out$testing$confusionMatrix)
                cat('Testing AUC: \n')
                print(out$testing$auc)
            }
        }
        if ('validation' %in% partitions) {
            out$validation <- get.model.accuracy(model, 'validation') 
            if (verbose) {
                cat('Validation accuracy =', get.annotation(1 - out$validation$misClassError, 
                                                         what = 'percent', parentheses = FALSE), '\n')
                cat('Validation confusion matrix: \n \n')
                print(out$validation$confusionMatrix)
                cat('Validation AUC: \n')
                print(out$validation$auc)
            }
        }
    }
    model$eval <- out
    return(model)
}


accuracy.rating <- function(value, accuracy_metric, predictor_var, reference_var) 
{
    # Returns a string that summarises the effectiveness of a predictor, 
    # based on the provided accuracy metric.
    # (Nb. Useful in Rmd files for verbally summarising results automatically.)
    
    if (value > 1 | value < 0) {
        # warning('accuracy.rating(): accuracy_metric must be in [0,1]')
        result <- 'Warning: accuracy.rating(): accuracy_metric must be in [0,1]'
    } else {
        if (value < 0.5) adject <- 'very poor'
        if (value >= 0.5) adject <- 'poor'
        if (value >= 0.6) adject <- 'fair'
        if (value >= 0.7) adject <- 'good'
        if (value >= 0.8) adject <- 'very good'
        if (value >= 0.9) adject <- 'excellent'
        result <- paste0('Using ', accuracy_metric, ', ', predictor_var, ' is a ', adject, ' predictor of ', reference_var)
    }
    return(result)
}


cluster.summary <- function(kclust, print = TRUE, replace.p_zero = TRUE) #, X = NULL
{
    
    # Summarises the results of fitting a clustering model.
    # Applies generic and custom metrics, depending on the model type.
    #
    # NOTES:
    #
    #    1. Only kmeans & kproto are currently supported.
    #
    #    2. Categorical variables must be passed as factors. This means that logical variables acquire 
    #       a cardinal range of [1,2] rather than [0,1]
    #
    #    3. This function expects kclust to have a $data member; for kproto this is intrinsic, but for 
    #       kmeans it must be explicitly added to kclust (which is usually passed as model$fit).
    #
    
    cs <- list()
    
    if (class(kclust) != 'kmeans' & class(kclust) != 'kproto') {
        warning(paste('WARNING: cluster.summary(): Class', class(kclust), 'not supported.'))
    }
    
    # k <- nrow(kclust$size)
    k <- NROW(kclust$size)
    
    if (class(kclust) == 'kmeans') {
        data.classes <- rep('numeric', k)
    } 
    if (class(kclust) == 'kproto') {
        data.classes <- sapply(kclust$data, class)
    }
    
    # sep <- '\n'  #: doesn't work!
    sep <- '; '
    
    # Overall summary ...
    
    res1 <- paste0('Results (clustering = ', class(kclust), ', k = ', k, '): ')
    if (class(kclust) == 'kproto') {
        res1 <- paste0(res1, 'lambda = ', cround(kclust$lambda, 4), sep)
    }
    if (class(kclust) == 'kmeans') {
        res1 <- paste0(res1, 'totss = ', cround(kclust$totss, 4), sep)
    }
    res1 <- paste0(res1, 'tot.withinss = ', cround(kclust$tot.withinss, 4), sep)
    res1 <- paste0(res1, 'iter = ', as.character(kclust$iter), '')
    cs$metrics <- res1
    
    if (print) {
        cat('\n')
        cat(res1, '\n')
    }
    
    # Properties of clusters: ...
    
    tbl1 <- NA
    if (class(kclust) == 'kmeans') {
        tbl1 <- data.frame(size = as.integer(kclust$size), withinss = kclust$withinss, betweenss = kclust$betweenss)
    }
    if (class(kclust) == 'kproto') {
        tbl1 <- data.frame(size = as.integer(kclust$size), withinss = kclust$withinss, 
                           norm = kclust$withinss / as.integer(kclust$size))
    }
    cs$properties <- tbl1
    
    if (print) {
        cat('\nCluster properties: \n')
        print(cs$properties)
    }

    # Cluster centres ...
    
    tbl2 <- kclust$centers
    row.names(tbl2) <- as.character(seq(1, k))
    cs$centres <- tbl2
    
    if (print) {
        cat('\nCluster centres: \n')
        print(cs$centres)
    }
    
    # Create a data.frame for applying column-wise stats:
    
    df <- cbind(kclust$data, kclust$cluster)
    names(df)[which(names(df) == 'kclust$cluster')] <- 'cluster'
    
    # if algo supports categorical data, convert factors to integer:
    
    if (class(kclust) %in% c('kproto')) {
        # classes.df <- sapply(df, class)
        for (i in seq(1, ncol(kclust$data))) {
            if ('factor' %in% data.classes[[i]]) {
                df[, i] <- as.integer(df[, i]) 
                #D: cat('Converted column', i, 'to integer \n')
            }
        }
    }

#     # kmeans class structure does not hold input data ...
#     
#     if (class(kclust) == 'kmeans') {
# #         df <- cbind(X, kclust$cluster)
#         kclust$data <- X
#         df <- cbind(kclust$data, kclust$cluster)
#     }
#     
#     # Statistics of data within clusters ...
#     
#     if (class(kclust) == 'kproto') {
#         tbl2a <- ddply(df, .(cluster), function(x) {
#             res <- as.numeric(colwise(min)(x))
#             names(res) <- names(kclust$data)
#             res
#         })
#     }
#     if (class(kclust) == 'kmeans') {
#         tbl2a <- ddply(df, .(cluster), function(x) {
#             res <- as.numeric(colwise(min)(x))
#             # names(res) <- names(X) #: big X (the datafeame of numeric predictors)
#             names(res) <- names(kclust$data)
#             res
#         })
#     }
#     
    
    # Statistics of data within clusters ...

    tbl2a <- ddply(df, .(cluster), function(x) {
        res <- as.numeric(colwise(min)(x))
        names(res) <- names(kclust$data)  #: omits 'cluster'
        res
    })

    cs$mins <- dplyr::select(tbl2a, -matches('cluster'))
    
    if (print) {
        cat('\nCluster minima: \n')
        print(cs$mins)
    }
    
    tbl2b <- ddply(df, .(cluster), function(x) {
        res <- as.numeric(colwise(max)(x))
        names(res) <- names(kclust$data)
        res
    })
    cs$maxs <- dplyr::select(tbl2b, -matches('cluster'))
    
    if (print) {
        cat('\nCluster maxima: \n')
        print(cs$maxs)
    }
    
    # For numeric variables, cluster means are the same as the cluster centres; for factors, 
    # they are non-integers indicating intermediacy between levels (e.g. 1.5 implies a 50:50 split between levels 1 & 2):

    tbl2a <- ddply(df, .(cluster), function(x) {
        res = as.numeric(colwise(mean)(x))
        names(res) = names(kclust$data)
        res
    })
    cs$means <- dplyr::select(tbl2a, -matches('cluster'))

    if (print) {
        cat('\nCluster means: \n')
        print(cs$means)
    }
    
    tbl3 <- ddply(df, .(cluster), function(x) {
        res <- as.numeric(colwise(sd)(x))
        names(res) <- names(kclust$data)
        res
    })
    cs$sdevs <- dplyr::select(tbl3, -matches('cluster'))
    
    if (print) {
        cat('\nCluster standard deviations: \n')
        print(cs$sdevs)
    }
    
    # p.values of cluster data statistics (a low p.value means the clusters are dissimilar with respect to the data variable):
    # Nb. This only makes sense for k = 2 !!! NEED an ANOVA-based multinomial method for k > 2 !!!!!!
    # res <- list()
    # for (i in seq(1, ncol(kclust$data))) {
    #     if ('factor' %in% data.classes[[i]]) {
    #         # compare cluster 2 against cluster 1:
    #         
    #         # cat('column', i, 'is a factor \n') #:D
    #         
    #         # if (nlevels(kclust$data[, i] == 2)) {
    #         #     # test two binomial distributions by comparing frequency of class 2 ("TRUE") in each cluster:
    #         #     res[[i]] <- binom.test(as.integer(kclust$data[kclust$cluster == 1, i]) == 2, sum(kclust$cluster == 1), 
    #         #                            sum(as.integer(kclust$data[kclust$cluster == 2, i]) == 2) / sum(kclust$cluster == 2))$p.value  
    #         # } else {
    #         #     d'oh!
    #         # }
    #         # cat('i =', i, '\n')
    #         # cat('mean of cluster 1 =', mean(kclust$data[kclust$cluster == 1, i], na.rm = TRUE), '\n')
    #         # cat('mean of cluster 2 =', mean(kclust$data[kclust$cluster == 2, i], na.rm = TRUE), '\n')
    #         
    #         observed <- as.integer(table(kclust$data[kclust$cluster == 2, i]))
    #         prob <- as.integer(table(kclust$data[kclust$cluster == 1, i])) / sum(kclust$cluster == 1)
    #         invisible(capture.output(res[[i]] <- EMT::multinomial.test(observed, prob)$p.value))
    #     } else {
    #         # cat('column', i, 'is not a factor \n') #: D
    #         x <- kclust$data[kclust$cluster == 2, i]
    #         y <- kclust$data[kclust$cluster == 1, i]
    #         
    #         # WARNING: t.test() can call stop() if variance too small! (Changing seed may help.)
    #         # cat('mean x =', mean(x, na.rm = TRUE), '\n') #: D
    #         # cat('mean y =', mean(y, na.rm = TRUE), '\n') #: D
    #         # res[[i]] <- 0.99999
    #         res[[i]] <- t.test(x, y)$p.value
    #         
    # TODO: try t.test.cluster()   
    #     }
    # }
    # res <- unlist(res)
    # names(res) <- names(kclust$data)
    # cs$p.values <- res
    # 
    # if (print) {
    #     cat('\nCluster p.values: \n')
    #     print(cs$p.values)
    # }
    # 
    # if (replace.p_zero) {
    #     cs$p.values[cs$p.values == 0] <- 9999
    # }

    return(cs)
}


plotPredictorStats <- function(x, y, xaxt = 'n', xlab = '', ylab, main, pch = 19, col = 4) {   
    # Plots a statistic of predictor performance (relative significance or power).

    plot(x = x, y = y, xaxt = xaxt, xlab = '', ylab = ylab, main = main, pch = pch, col = col) 
    lines(x = x, y = y, col = 2)  
    axis(side = 1, at = x, mlabs = xlab, tck = 0, las = 2)
}


normalise <- function(x, sides = 'either') {
    
    # Rescales a vector according to the range of the positive side, negative side, 
    # the greater of the two, or the sum of the two.
    
    if (sides == 'either') {
        x <- x / max(abs(range(x, na.rm = TRUE)))
    }
}


getSimpleClasses <- function(df) {
    
    # Returns a vector of (simple) column classes (e.g. c('ordered', 'factor') -> 'factor').
    
    # Nb. Includes a kludge in case class() on an ordered factor does not always return c('ordered', 'factor')
    
    class.list <- sapply(df, class)
    l <- length(class.list)
    class.vector <- character(l)
    for (i in seq(1, l)) {
        class.vector[i] <- class.list[[i]][length(class.list[[i]])]
        if (class.vector[i] == 'ordered') class.vector[i] <- 'factor'
    }
    return(class.vector)
}


vector.indices <- function(x, cvec) {
    # Returns indices of a named vector or data frame that match the input names.
    
    return(which(names(x) %in% cvec))
}


# NOTES for next commit:
#   added random number generator seed to parameter list
#   captured params in model$params (list)
#
setupModel <- function(indata, algo_code, algo_name, 
                       response_col, predictor_cols = NULL, tag_col = NULL, 
                       response_levels = c(0, 1, NA), 
                       model_name = NULL, subset_name = NULL, 
                       partitioning = c(0.7, 0.3), nfolds = 1, seed = 7, 
                       transforms = NULL, response_type = 'NONE', 
                       boolToFactors = TRUE, na_response.rm = TRUE, 
                       na_predictor.rm = TRUE, invalid.rm = TRUE, 
                       verbose = FALSE) 
{
    # Returns a list that describes an ML model and contains training data.
    #
    # Parameters:
    #
    #   indata              Input data frame (with or without response column); can include.
    #                         columns not used in model.
    #   algo_code           The name by which the algorithm is called in code.
    #   algo_name           The common name of the algorithm.
    #   response_type       One of: 
    #                           'logical'       domain = (FALSE, TRUE, NA) 
    #                           'boolean'       domain = (0, 1, NA)
    #                           'categorical'   domain = user-specified
    #                           'continuous'    domain = real numbers. 
    #                       Must be consistent with algo_code.
    #   response_col        The name of the column containing the response 
    #                         or a vector of response values (if categorical must be passed a factor).
    #   predictor_cols      The names of the columns for predicting the response 
    #                         or NULL (meaning: use all columns in indata as predictors).
    #   response_levels     The values that cover the response variable (must all be the same type): 
    #                           the 1st corresponds to FALSE
    #                           the 2nd corresponds to TRUE
    #                           the 3rd corresponds to NA (instances will be converted to NA).
    #   subset_name         The name of the data subset used (if any); informational only.
    #   transforms          A vector of predefined transform names (to be applied to data).
    #   partitioning        A vector of 2 proportions (summing to 1) for partitioning train/test data,
    #                       or 1 proportion (< 1) for partitioning train/test data,
    #                       or 3 proportions for partitioning train/validate/test data, 
    #                       or 0 for no partitioning.
    #   nfolds              Number of folds for cross-validation (1 for no cross-validation).
    #   seed                Random number generator seed.
    #   boolToFactors       Flag for converting logical columns to factors.
    #   na_predictor.rm     Flag for removing predictor rows with NA's.
    #   na_response.rm      Flag for removing response rows with NA's.
    #   invalid.rm          Flag for removing rows with invalid data.
    #   verbose             Flag for output to console.
    #
    # Structure of return variable:
    #
    #   "what am I?" flag:
    #       model$class                     = 'encmodel' (encapsulated model)
    #
    #   echoes of input data, parameters:
    #       model$algo_code           
    #       model$algo_name           
    #       model$response_type       
    #       model$response_col        
    #       model$predictor_cols      
    #       model$subset_name         
    #       model$params$transforms
    #       model$params$partitioning
    #       model$params$nfolds
    #       model$params$seed
    #       model$params$boolToFactors
    #       model$params$na_response.rm
    #       model$params$invalid.rm
    #
    #   output:
    #       model$mdat                      the data frame containing the predictor data (unpartitioned) 
    #       model$mlabs                     the vector containing the response data (unpartitioned)
    #       model$mtags                     the vector containing the tags (unpartitioned)
    #       model$training                  logical vector of training partition membership 
    #       model$testing                   logical vector of testing partition membership (optional) 
    #       model$validation                logical vector of validation partition membership (optional) 
    #       model$invalid                   logical vector of invalid rows
    #       mode$fit                        the trained model
    #
    #
    # Notes:
    #
    #    1. The `invalid.rm` flag has not yet been fully implemented. Currently, it can be used for 
    #       detecting illegal values in a logical or boolean response vector. In future, it will also  
    #       detect out-of-range values for categorical and continuous responses.
    #
    # ToDo:
    #
    #    1. column-level control of transforms
    #       - currently, a specified transform gets applied to all predictor columns
    #    2. zero management before log10() 
    #       - warn; apply default option; present options
    #       - options:
    #           - offset by 0.01 * smallest nonzero value 
    #           - drop row
    #           - use sqrt instead
    #    3. alternative range compression methods
    #       - sqrt (with negative-handling)
    #    4. automatic management of dynamic range using heuristics to select method
    #    5. (optionally) exclude invalid rows before partitioning data set
    #
    
    if (is.null(transforms)) {
        transforms = c('NONE')
    }
    valid.transforms = c('normalise', 'log10')
    
    model <- list()
    model$class <- 'encmodel'   # deprecated
    model$name <- ifelse(is.null(model_name), 'NONE', model_name)
    model$algo_code <- algo_code
    model$algo_name <- algo_name
    
    if (response_type %in% c('logical', 'boolean', 'categorical', 'continuous')) {
        model$response_type <- response_type
    } else {
        warning(paste0("setupModel(): response_type not in c('logical', 'boolean', 'categorical', 'continuous')."))
    }
    
    model$params <- list()
    model$params$transforms <- transforms
    model$params$partitioning <- partitioning
    model$params$nfolds <- nfolds
    model$params$seed <- seed
    model$params$boolToFactors <- boolToFactors
    model$params$na_response.rm <- na_response.rm
    model$params$invalid.rm <- invalid.rm

    set.seed(seed)
    
    # model$predictor_cols will be specified columns, entire input dataset:
    if (is.null(predictor_cols)) {
        model$mdat <- indata
        model$predictor_cols <- names(indata)
    } else {
        model$mdat <- indata[, which(names(indata) %in% predictor_cols)]
        model$predictor_cols <- predictor_cols
    }
    
    if (class(response_col) == 'character') {  
        # the name of the response column was passed
        model$response_col <- response_col
        model$mlabs <- indata[, which(names(indata) == model$response_col)]
    } else {                                  
        # a vector of responses was passed
        model$mlabs <- response_col
        model$response_col <- tag_col.external_
    }
    
    # transform (if required) and validate response type, replace NA flags with NA ----------

    rows.invalid <- rep(FALSE, nrow(model$mdat))
    
    if (response_type == 'logical') {
        if (class(model$mlabs) == 'logical') {
            # only (TRUE, FALSE, NA) are possible: nothing to do!
            new.mlabs <- model$mlabs   # dummy statement
        } else {
            if (class(model$mlabs) == 'integer') {
                # convert 0 to FALSE, 1 to TRUE:
                new.mlabs <- (model$mlabs == response_levels[2])  
                new.mlabs[model$mlabs == response_levels[3]] <- NA  # overwrite FALSE in missing values with NA 
                model$mlabs <- new.mlabs
            } else {
                warning('setupModel(): Unsupported class for response_type.')
            }
        }
    } else {
        if (response_type == 'boolean') {
            if (class(model$mlabs) == 'integer') {
                # if the NA flag is other than NA itself, overwrite flagged values with NA:
                if (!is.na(response_levels[3])) {
                    # TODO: test on zero-length string, etc.!
                    model$mlabs[model$mlabs == response_levels[3]] <- NA
                    #D: table(model$mlabs, useNA = 'always')
                }
                # only (0, 1, NA) are allowed:
                # rows.invalid <- setdiff(which(!(model$mlabs %in% c(0, 1))), which(is.na(model$mlabs)))
                rows.invalid <- !(model$mlabs %in% c(0, 1))  # check: does this preserve NAs ?
            } else {
                if (class(model$mlabs) == 'logical') {
                    # convert FALSE to 0, TRUE to 1:
                    new.mlabs <- ifelse(model$mlabs == response_levels[2], 1, 0)  
                    new.mlabs[model$mlabs == response_levels[3]] <- NA  # overwrite FALSE in missing values with NA 
                    model$mlabs <- new.mlabs
                } else {
                    warning('setupModel(): Unsupported class for response_type.')
                }
            }
        }
    }
    
    # flag invalid rows in model object ------------
    
    # if (exists(rows.invalid)) {
    #     model$invalid <- (row.names(model$mat) %in% rows.invalid)
    # }
    model$invalid <- rows.invalid
    
    if (is.null(tag_col)) {
        # default: tag column is row numbers:
        model$tag_col <- tag_col.row_numbers_
        model$mtags <- row.names(indata)
    } else {
        if (class(tag_col) == 'character') {  
            # the name of the tag column was passed:
            model$tag_col <- tag_col
            model$mtags <- indata[, which(names(indata) == model$tag_col)]
        } else {                             
            # a vector of tags was passed:
            model$tag_col <- tag_col.external_
            model$mtags <- tag_col
        }
    }

    # apply transforms ----------
    
    model$mdat.raw <- model$mdat        # keep an un-transformed version of the input data (e.g. for plotting)
    model$subset_name <- ifelse(is.null(subset_name), 'NONE', subset_name)
    model$transforms <- list()
    
    if (NROW(model$mlabs) != nrow(model$mdat)) {
        warning(paste0('setupModel(): before transforms: unequal row count: mdata = ', nrow(model$mdat), 
                       ', mlabs = ', NROW(model$mlabs)))
    }
    
    if (transforms[1] == 'NONE') {
        model$transforms <- 'NONE'
    } else {
        classes <- getSimpleClasses(model$mdat)
        for (i in 1:ncol(model$mdat)) {
            if (classes[i][1] == 'integer') {
                model$mdat[, i] <- as.numeric(model$mdat[, i])
                cat('converted', names(model$mdat)[i], 'to integer \n')
            }
            if (classes[i][1] == 'numeric') {
                if ('log10' %in% transforms) {
                    model$mdat[, i] <- log10(model$mdat[, i])
                    cat('converted', names(model$mdat)[i], 'to log10 \n')
                }
                if ('normalise' %in% transforms) {
                    model$mdat[, i] <- normalise(model$mdat[, i])
                    cat('normalised', names(model$mdat)[i], ' \n')
                }
            }
            if (classes[i][1] == 'logical' && boolToFactors) {
                model$mdat[, i] <- as.factor(model$mdat[, i])
            }
            
            if (NROW(model$mlabs) != nrow(model$mdat)) {
                warning(paste0('setupModel(): after transforms: unequal row count: mdata = ', 
                               nrow(model$mdat), ', mlabs = ', NROW(model$mlabs)))
            }
        }
        model$transforms <- strsplit(transforms)
    }
    
    
    # TODO: optionally apply validations to non-logical columns based on rules passed in parameter list ------
    

    # optionally remove invalid rows from model object ------------
    
    if (sum(model$invalid) > 0) {
        if (invalid.rm) {
            rows.keep <- !model$invalid
            model$mdat <- model$mdat[rows.keep, ]
            model$mlabs <- model$mlabs[rows.keep]
            model$mtags <- model$mtags[rows.keep]
        }
    }

    # optionally remove NA response rows from model object ------------
    
    if (na_response.rm) {
        if (sum(is.na(model$mlabs)) > 0) {
            rows.keep <- !is.na(model$mlabs)
            model$mdat <- model$mdat[rows.keep, ]
            model$mlabs <- model$mlabs[rows.keep]
            model$mtags <- model$mtags[rows.keep]
        }
    }
    
    # optionally remove NA predictor rows from model object ------------
    
    if (na_predictor.rm) {
        model <- drop_na(model)
    }
    
    
    # partition data by 2 (training, testing) or 3  (training, validation, testing) ----------
    
    # nSamples <- NROW(model$mlabs)
    nAll <- nrow(model$mdat)
    if (length(partitioning) == 1) {
        if (partitioning == 0) {
            model$training <- rep(TRUE, nSamples)
        } else {
            training <- sample(row.names(model$mdat), size = nAll * partitioning, replace = FALSE)
            model$training <- (row.names(model$mdat) %in% training)
            model$testing <- !model$training
        }
    } else {
        if (length(partitioning) == 2) {
            training <- sample(row.names(model$mdat), size = nAll * partitioning[1], replace = FALSE)
            model$training <- (row.names(model$mdat) %in% training)
            model$testing <- !model$training
        } else {
            training <- sample(row.names(model$mdat), size = nAll * partitioning[1], replace = FALSE)
            testorval <- setdiff(row.names(model$mdat), model$training)
            validation <- sample(testorval, 
                                 size = length(testorval) * partitioning[2] / (partitioning[2] + partitioning[3]), 
                                 replace = FALSE)
            testing <- setdiff(testorval, model$validation)
            model$training <- (row.names(model$mdat) %in% training)
            model$validation <- (row.names(model$mdat) %in% validation)
            model$testing <- (row.names(model$mdat) %in% testing)
        }
    }
    
    
    # set up for (optional) cross-validation ------------
    
    # TODO: replace $cv.auc with a list of lists (metric specified by user in parameter list)
    
    if (nfolds > 1) {
        model$cv.fit <- list()
        model$cv.yhat <- list()
        model$cv.yhat.class <- list()
        model$cv.auc <- list()
        model$folds <- createFolds(model$mlabs[model$training], k = nfolds)
    }
    
    if (verbose) {
        cat('Model', model$name, 'has', sum(model$training), 'training samples,', 
            sum(model$testing), 'testing samples.', nrow(titanic) - nrow(model$mdat), 
            'samples were removed due to missing or invalid data. \n')
    }
    
    return(model)
}


drop_na <- function(model) {
    
    # Removes data rows with NAs from model (including corresponding label rows).
    
    df <- model$mdat
    if (!is.null(model$mlabs)) {
        df$mlabs <- model$mlabs
    }
    if (!is.null(model$mtags)) {
        df$mtags <- model$mtags
    }
    
    df <- na.omit(df)
    
    if (!is.null(df$mlabs)) {
        model$mlabs <- df$mlabs
        df <- subset(df, select = -mlabs)
    }
    if (!is.null(df$mtags)) {
        model$mtags <- df$mtags
        df <- subset(df, select = -mtags)
    }
    
    model$mdat <- df
    model$mdat.raw <- model$mdat
    
    return(model)
}


Accuracy <- function(TP, FP, TN, FN)
{
    total <- TP + FP + TN + FN
    A <- 1 - (FP + FN) / total
    return(A)
}    


Kappa <- function(TP, FP, TN, FN)
{
    A <- Accuracy(TP, FP, TN, FN)
    total <- TP + FP + TN + FN
    Pe <- ((TP + FP) * (TP + FN) + (TN + FN) * (FP + TN)) / total ^ 2
    K <- (A - Pe) / (1 - Pe)
    return(K)
}    


# TESTS:
#
# TP <- 4; TN <- 4; FP <- 1; FN <- 1
# Accuracy(TP, FP, TN, FN)
# Kappa(TP, FP, TN, FN)  #: 'kappa' more conservative that 'accuracy'
# 
# TP <- 5; TN <- 5; FP <- 0; FN <- 0
# Accuracy(TP, FP, TN, FN)
# Kappa(TP, FP, TN, FN)


misClassError <- function(yhat.class, mlabs) 
{
    return(mean(yhat.class != mlabs))
}


trainModel <- function(model, partition = NULL, FUN, ...) 
{
    # NOT FINISHED!
    #
    #   Invokes a specified ML algorith and appends the output object to the model 
    #                           object 
    
    if (is.null(partition)) {
        mdat <- model$mdat
        if (!is.null(model$mlabs)) mlabs <- model$mlabs
    } else {
        if (partition == 'training') {
            mdat <- model$mdat[training, ]
            if (!is.null(model$mlabs)) mlabs <- model$mlabs[training]
        }
        if (partition == 'testing') {
            mdat <- model$mdat[testing, ]
            if (!is.null(model$mlabs)) mlabs <- model$mlabs[training]
        }
        if (partition == 'validation') {
            mdat <- model$mdat[validation, ]
            if (!is.null(model$mlabs)) mlabs <- model$mlabs[validation]
        }
    }
    
    fit <- FUN(mdat, mlabs, ...)  # how is this going to work?!
    
    modedl$fit <- fit
    return(model)
}


# TODO: MUST specify validation, if any, at time of partitioning !!!!!!!!!!!
# TODO: MUST assign testing, validation explicitly !!!!!!!!!



