# File:       eml_framework_v0.6.R
# Authors:    Allen H. Nugent, 2018+
# Last edit:  2018-06-13
# Last test:  2018-06-13
# Purpose:    Demonstrates a framework for low-level management of machine learning practice.
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#


library(caret)
library(randomForest)
library(plyr)
library(dplyr)
library(Hmisc)          # rcorr()
library(pROC)           # plot.roc()

# source(paste(folder.lib, 'cleanly_ex.R', sep = '/'))
# source(paste(folder.lib, 'strfns.R', sep = '/'))
# source(paste(folder.lib, 'mlfns.R', sep = '/'))
# source(paste(folder.lib, 'numfns.R', sep = '/'))

file.dat <- 'titanic.csv'
file.datwe <- 'titanicWithEthnicity.csv'


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Preparation                                                                           ===========
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Local Functions ---------------------------------------------------------------------------------

addPredictors <- function(model) 
{
    
    cat('addPredictors(): Null effect! This function is a placeholder only. \n',
        'Actual functionality will be problem-specific. \n')
}


# Load data ---------------------------------------------------------------------------------------

# titanic <- read.csv(paste(folder.dat, file.dat, sep = '/'))
# titanicwe <- read.csv(paste(folder.dat, file.datwe, sep = '/'))
titanic <- read.csv(file.dat)
titanicwe <- read.csv(file.datwe)
setdiff(names(titanicwe), names(titanic))


# Preprocessing Common to All Algos ---------------------------------------------------------------

# drop redundant 'X' column:

if (NROW(which(names(titanic) == 'X')) > 0)  
    titanic <- titanic[, -which(names(titanic) == 'X')]       
if (NROW(which(names(titanicwe) == 'X')) > 0)  
    titanicwe <- titanicwe[, -which(names(titanicwe) == 'X')]


# Data profiling ----------------------------------------------------------------------------------

# check for missing data:

sapply(titanicwe, function(x) sum(is.na(x)))
summary(titanicwe)




# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Model Naming                                                                           ==========
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# Nb. The following scheme is not prescriptive!
#
# A model name is composed of:
#
#   "A" plus an algorithm number (with optional letter for a variant)
#
#   "P" plus a number indicating the predictor set
#
#   "L" plus a number indicating the label vector
#
# The algorithm number represents the highest level modelling decision.
# 
# Model variants include variations in preprocessing and parameter values.
#
# Labelling schemes come into play when applying supervised learning to a problem in which the data are not 
# actually labelled, and an arbitrary proxy must be used instead. 


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Model Definition                                                                      ===========
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#
# The `model` object incorporates the following descriptive fields, some or all of which 
# might be captured in a naming scheme:
#
# model$name                the name of the model object (without the 'model.'prefix)
# model$algo_code           the name by which the algorithm is called in code
# model$algo_name           the common name of the algorithm
# model$response_col        the name of the column containing the response 
# model$predictor_cols      the names of the columns for predicting the response
# model$subset_name         the name of the data subset used (if any)
# model$mdat                the data frame containing the predictor data (unpartitioned) 
# model$mlabs               the vector containing the response data (unpartitioned)
# model$mtags               the vector containing the row identifiers (unpartitioned)


# Algorithms --------------------------------------------------------------------------------------

A1.code <- 'glm'  # binomial logit'
A1.name <- 'logistic regression'

A2.code <- 'naiveBayes'
A2.name <- 'naive Bayes classifier'

A3.code <- 'randomForest'
A3.name <- 'random forest classifier'


# Labelling ---------------------------------------------------------------------------------------

# In the Titanic data set the outcome is unambiguous, so only one labelling scheme is required:

L1 <- 'Survived'


# Predictor Sets ----------------------------------------------------------------------------------

# The choice of predictor set may be governed by the alogorithm (e.g. continuous variables only). 
# Variants include predictor sets enhanced through feature engineering. 

P1 <- c('Pclass', 'Sex', 'Age')
P2 <- c('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare')
P3 <- c(P2, 'Ethnicity') #: titanicwe dataset only

# Optional feature vetting:
numeric_feature_cols <- names(titanic)[sapply(titanic, is.numeric)]
correlation_matrix <- rcorr(as.matrix(titanic[, vector.indices(titanic, numeric_feature_cols)]))
fc <- findCorrelation(correlation_matrix$r, names = TRUE, verbose = FALSE)
se <- string.elements(as.character(fc), print = TRUE)

# Optional feature engineering:
titanic3 <- addPredictors(titanicwe)  # TODO: code this function!
P4 <- c(P3, setdiff(names(titanic3), names(titanicwe)))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Model Initialisation, Training, Evaluation                                             ==========
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Algorithm A1: logistic regression ===============================================================

# Initialisation ------------------------------------------------------

# set.seed(7) #: moved into setupModel() with default param value
model.A1P1L1 <- setupModel(indata = titanic, algo_code = A1.code, algo_name = A1.name, 
                           response_col = L1, predictor_cols = P1, tag_col = 'PassengerId', 
                           response_levels = c(0, 1, -2000), 
                           model_name = 'A1P1L1', subset_name = 'NONE', 
                           partitioning = c(0.7, 0.3), nfolds = 1, 
                           response_type = 'boolean', boolToFactors = TRUE)

cat('Model', model.A1P1L1$name, 'has', sum(model.A1P1L1$training), 'training samples,', 
    sum(model.A1P1L1$testing), 'testing samples.', nrow(titanic) - nrow(model.A1P1L1$mdat), 
    'samples were removed due to missing or invalid data.')


# Data Transformations ---------------------------------------

# Here is where we could transform the data using a custom function 
# (defined in this script or in a project-specific script):  

# model.A1P1L1 <- transform.A1P1(model.A1P1L1) 


# Model Training & Evaluation --------------------------------

# train model:

model.A1P1L1$fit <- glm(mlabs ~., family = binomial(link = 'logit'), 
                          data = mutate(model.A1P1L1$mdat[model.A1P1L1$training, ], 
                                        mlabs = as.integer(model.A1P1L1$mlabs[model.A1P1L1$training])))

# compute model predictions:

model.A1P1L1 <- getModelPredictions(model.A1P1L1)
model.A1P1L1 <- getModelClasses(model.A1P1L1, threshold = 0.5)

# evaluate model:

model.A1P1L1<- evaluateModel(model.A1P1L1, c('training', 'testing'))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Analysis of Results                                                                ==============
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# Model evaluations ------------------------

cat('Predicted survival rate =', mean(model.A1P1L1$yhat.class), '\n',
    'Actual survival rate =', mean(model.A1P1L1$mlabs), '\n')

print(model.A1P1L1$eval)

par(mar = c(4, 4, 4, 4) + .1)
plot.roc(model.A1P1L1$mlabs[model.A1P1L1$training], model.A1P1L1$yhat[model.A1P1L1$training], 
         percent = TRUE, add = FALSE, col = "blue", lwd = 3, 
         main = paste0("ROC Curve for Model ", model.A1P1L1$name, ': Train'))
rect(100, 0, 0, 100, border = 'gray')
plot.roc(model.A1P1L1$mlabs[model.A1P1L1$testing], model.A1P1L1$yhat[model.A1P1L1$testing], 
         percent = TRUE, add = FALSE, col = "blue", lwd = 3, 
         main = paste0("ROC Curve for Model ", model.A1P1L1$name, ': Test'))
rect(100, 0, 0, 100, border = 'gray')
par(mar = c(4, 3, 1, 1))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Save / Retrieve Models                                                              =============
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# save fitted model, data, evaluation results ...
saveRDS(model.A1P1L1, file = paste(folder.dat, 'model.A1P1L1', sep = '/'))

# retrieve fitted model, data, evaluation results ...
my_model <- readRDS(file = paste(folder.dat, 'model.A1P1L1', sep = '/'))


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Model Experimentation                                                              ==============
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


# Predictor Set P2 with Algo A1 ===========================================================================

model.A1P2L1 <- setupModel(indata = titanic, algo_code = A1.code, algo_name = A1.name, 
                           response_col = L1, predictor_cols = P2, tag_col = 'PassengerId', 
                           response_levels = c(0, 1, -2000), 
                           model_name = 'A1P2L1', subset_name = 'NONE', 
                           partitioning = c(0.7, 0.3), nfolds = 1, seed = 77, 
                           response_type = 'boolean', boolToFactors = TRUE, verbose = TRUE)

model.A1P2L1$fit <- glm(mlabs ~., family = binomial(link = 'logit'), 
                        data = mutate(model.A1P2L1$mdat[model.A1P2L1$training, ], 
                                      mlabs = as.integer(model.A1P2L1$mlabs[model.A1P2L1$training])))

model.A1P2L1 <- getModelPredictions(model.A1P2L1)
model.A1P2L1 <- getModelClasses(model.A1P2L1, threshold = 0.5)
model.A1P2L1 <- evaluateModel(model.A1P2L1, c('training', 'testing'))
print(model.A1P2L1$eval)


# Predictor Set P3 with Algo A1 (logreg) ==================================================================

model.A1P3L1 <- setupModel(indata = titanicwe, algo_code = A1.code, algo_name = A1.name, 
                           response_col = L1, predictor_cols = P3, tag_col = 'PassengerId', 
                           response_levels = c(0, 1, -2000), 
                           model_name = 'A1P3L1', subset_name = 'NONE', 
                           partitioning = c(0.7, 0.3), nfolds = 1, seed = 777, 
                           response_type = 'boolean', boolToFactors = TRUE, verbose = TRUE)

model.A1P3L1$fit <- glm(mlabs ~., family = binomial(link = 'logit'), 
                        data = mutate(model.A1P3L1$mdat[model.A1P3L1$training, ], 
                                      mlabs = as.integer(model.A1P3L1$mlabs[model.A1P3L1$training])))

model.A1P3L1 <- getModelPredictions(model.A1P3L1)
model.A1P3L1 <- getModelClasses(model.A1P3L1, threshold = 0.5)
model.A1P3L1 <- evaluateModel(model.A1P3L1, c('training', 'testing'))
print(model.A1P3L1$eval$testing$auc)  # print auc only, for testing dataset only



# Algorithm A2: naiveBayes =============================================================================

model.A2P1L1 <- setupModel(indata = titanic, algo_code = A2.code, algo_name = A2.name, 
                           response_col = L1, predictor_cols = P1, tag_col = 'PassengerId', 
                           response_levels = c(0, 1, -2000), 
                           model_name = 'A2P1L1', subset_name = 'NONE', 
                           partitioning = c(0.7, 0.3), nfolds = 1, 
                           response_type = 'logical', boolToFactors = TRUE, verbose = TRUE)

model.A2P1L1$fit <- naiveBayes(mlabs ~., 
                               data = mutate(model.A2P1L1$mdat, 
                                             mlabs = as.logical(model.A2P1L1$mlabs)),
                               subset = model.A2P1L1$training)

model.A2P1L1 <- getModelPredictions(model.A2P1L1)
model.A2P1L1 <- getModelClasses(model.A2P1L1, threshold = 0.5)
model.A2P1L1<- evaluateModel(model.A2P1L1, c('training', 'testing'), verbose = TRUE)  # prints all supported evaluations for algo

par(mar = c(4, 4, 4, 4) + .1)
plot.roc(model.A2P1L1$mlabs[model.A2P1L1$testing], model.A2P1L1$yhat[model.A2P1L1$testing], 
         percent = TRUE, add = FALSE, col = "blue", lwd = 3, 
         main = paste0("ROC Curve for Model ", model.A2P1L1$name, ': Test'))
rect(100, 0, 0, 100, border = 'gray')
par(mar = c(4, 3, 1, 1))




# Predictor Set P3 with Algo A3 ===========================================================================

model.A3P3L1 <- setupModel(indata = titanicwe, algo_code = A3.code, algo_name = A3.name, 
                           response_col = L1, predictor_cols = P3, tag_col = 'PassengerId', 
                           response_levels = c(0, 1, -2000), 
                           model_name = 'A3P3L1', subset_name = 'NONE', 
                           partitioning = c(0.7, 0.3), nfolds = 1, seed = 777, 
                           response_type = 'boolean', boolToFactors = TRUE, verbose = TRUE)

model.A3P3L1$fit <- randomForest(x = model.A3P3L1$mdat[model.A3P3L1$training, ], 
                                 y = as.factor(model.A3P3L1$mlabs[model.A3P3L1$training]))

model.A3P3L1 <- getModelPredictions(model.A3P3L1)
#E: model.A3P3L1 <- getModelClasses(model.A3P3L1, threshold = 0.5)
model.A3P3L1 <- evaluateModel(model.A3P3L1, c('training', 'testing'))
print(model.A3P3L1$eval)



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Using Cross-Validation                                                              =============
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

kfolds = 10
model.A1P1L1cv <- setupModel(indata = titanic, algo_code = A1.code, algo_name = A1.name, 
           response_col = L1, predictor_cols = P1, tag_col = 'PassengerId', 
           response_levels = c(0, 1, -2000), 
           model_name = 'A1P1L1cv', subset_name = 'NONE', 
           partitioning = c(0.7, 0.3), nfolds = kfolds, 
           response_type = 'boolean', boolToFactors = TRUE)

folds <- createFolds(model.A1P1L1cv$mlabs[model.A1P1L1cv$training], k = kfolds)
eval.A1P1L1cv <- list()

# NOT WORKING .......................................................................................
for (fold in 1:kfolds) {
    cat('fold', fold, '...\n')
    model.A1P1L1cv$cv.fit[[fold]] <- glm(mlabs ~., family = binomial(link = 'logit'),
                                          data = mutate(model.A1P1L1$mdat[folds[[fold]], ], 
                                                        mlabs = model.A1P1L1$mlabs[folds[[fold]]]))
    
    model.A1P1L1$cv.yhat[[fold]] <- getModelPredictions(model.A1P1L1$cv.fit[[fold]],
                                                           model.A1P1L1$mdat[folds[[fold]], ])
    model.A1P1L1$cv.yhat.class[[fold]] <- getModelClasses(model.A1P1L1$cv.yhat[[fold]],
                                                               model.A1P1L1$response_type)

    eval.A1P1L1cv[[fold]] <- evaluateModel(model.A1P1L1cv, folds[[fold]], verbose = FALSE,
                                               evals = 'auc')$eval$auc
}
#....................................................................................................




