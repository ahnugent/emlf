# Script:       statsfns.R
# Authors:      Allen H Nugent
# Created:      2016-09-30
# Last edit:    2018-05-29
# Last test:    
#
# Purpose:      Library of higher-level statistics utilities and data structures.
#
# Contents:
#
#   binsort             Bins data as for a histogram and returns in a list.
#
#   censoring           Tests a vector for left/right censoring. (Nb. Used as a clipping detector by signalysis.R).
#
#   compare             Compares two values: difference (plus %), significance.
#
#   confusion           Computes stats from confusion matrix.
#
#   datstats            Computes descriptive statistics of a vector, with handling of invalid data
#                       and optional ndig of output.
#
#   descrstats          Computes descriptive statistics.
#
#   extractMatrix       Extracts a matrix from a data frame, ignoring non-numeric columns.
#
#   get.ndim.stats      Computes number of rows in a vector, with optional ndig.
#
#   getTail             Returns the subset of a distribution whose values are on one side 
#                       or the other of a threshold fraction of the extreme value.
#
#   hist.discrete       Plots a discrete histogram by counts or frequencies.
#
#   ifsignif            Returns a string depending on result of comparing a test 
#                       statistic to a significance level.
#
#   Mode                Returns the mode of a distribution.
#   modex               DEPRECATED: Returns the mode of a distribution.
#
#   obfuscate           Returns a vector of anonymised identifiers randomly mapped to the input vector.
#
#   obfuscator          Generates a lookup table of input identifiers and randomly mapped output identifiers 
#                       appropriate to the input range or with specified number of digits.
#
#   randomseed          Generates a random seed for set.seed().
#
#   spiking             Tests a vector for spikes (i.e. local maxima in a binned distribution). 
#                       Returns bin indices.
#


# NFG: loadPackages(c("e1071", "diptest", "Hmisc", "quantmod"))
if (!require("e1071")) install.packages("e1071")
if (!require("diptest")) install.packages("diptest")
if (!require("Hmisc")) install.packages("Hmisc")
if (!require("quantmod")) install.packages("quantmod")
# if (!require("GGally")) install.packages("GGally")

# two commonly-used names for library folder variable: folder.lib & path.lib ...
if (exists('folder.lib')) {
    statsfns.folder.lib <- folder.lib
} else {
    if (exists('path.lib')) {
        statsfns.folder.lib <- path.lib
    }
}
if (statsfns.folder.lib != '') {
    source(paste(statsfns.folder.lib, "numfns.R", sep = "/"))
} else {
    warning('statsfns.R: Cannot find library folder.')
    # this alledgedly works outside RStudio:
    # script.dir <- dirname(sys.frame(1)$ofile)
}


censoring <- function(x, method = "max", tail.size = 0.05, window.width = 0.1, threshold = 1.5) 
{    
    # Tests a vector for left/right censoring. Returns each cut-off value (or NA).
    #
    # Method:   For each side of the distribution, zooms in on an extreme quantile (the 'tail'),
    #           puts the samples into discrete bins, and employs one of several methods 
    #           to compare the count in the terminal bin (i.e. the tip of the tail) 
    #           to a prediction based on a windowed series of adjacent bins. 
    #           If the tail-tip is higher than the any bin in the adjacent 
    #           portion, by a designated factor, clipping is true.
    #
    # Parameters:
    #
    #   method          how to define an outlier:
    #
    #                       "max"       for each bin, compare count of extreme value with counts 
    #                                   of other values in tail.
    #
    #                       "prob"      for each tail, compute probability of table[?] occurrences 
    #                                   of the extreme value -- TODO: NOT YET CODED !
    #
    #                       "trend"     for each tail, for a linear model to tails, then compare the
    #                                   predicted and actual values.
    #
    #   tail.size       size of first/last quantiles (determines number of samples in left/right tails).
    #
    #   window.width    proportion of tail bins to use for comparison (determines number of adjacent 
    #                   tail points to compare with tail-end counts).
    #
    #   threshold       multiplicative factor for comparison of tail-end to window (determines how 
    #                   much higher the tail-end peak must be than the highest peak in the window).
    #
    # Notes:
    #
    #    1. For dynamic signals, sensitivity and selectivity can be strongly dependent on the arguments, 
    #       especially for small sample sizes (e.g. N=O(100) samples in each tail).
    #
    #    2. If threshold = 1, comparison will fail; therefore, is adjusted in code.
    #

    out <- rep(NA, 2)
    
    if (threshold == 1) { threshold <- 1.01 }  # round-off error (?) causes test to fail
        
    numquantiles <- floor(1 / tail.size)
    # bins <- cut2(x, g = numquantiles + 1)
    bins <- cut2(x, g = numquantiles)
    tail.left <- x[bins == levels(bins)[1]]
    tail.right <- x[bins == levels(bins)[numquantiles]]
    
    # zoom in on a fraction of the tail ...
    
    table.left <- table(tail.left)      
    L.left <- length(table.left) 
    Lw.left <- ceiling(L.left * window.width)
    table.left <- table.left[1:Lw.left]
    L.left <- length(table.left) 
    end.left <- min(x)
    
    table.right <- table(tail.right)
    L.right <- length(table.right)
    Lw.right <- ceiling(L.right * window.width)
    table.right <- table.right[(L.right - Lw.right + 1):L.right]
    L.right <- length(table.right)
    end.right <- max(x)
    
    # browser()
    
    if (method == "max") {
        # for each tail, compare count of extreme value with counts of other values in tail ...
        # (Nb. it may be necessary to manage round-off error by binning the tails!)
        
        if (table.left[1] > threshold * max(table.left[2:L.left])) {
            out[1] <- end.left
        }
        if (table.right[L.right] > threshold * max(table.right[1:(L.right - 1)])) {
            out[2] <- end.right
        }
    } else if (method == "prob") {
        # for each tail, compute probability of table[?] occurrences of the extreme value ...
        
        print(paste0("censoring(): Argument not yet supported: method = '", method, "' -- using 'trend'"))
        
    } else if (method == "trend") {
        # for each tail, for a linear model to all bins except the extreme one, then compare the
        # predicted and actual values of the tail-end count ...
        
        i.tail <- 2:(L.left)
        x.tail <- as.numeric(names(table.left)[i.tail])
        y.tail <- table.left[i.tail]
        trend <- lm(y.tail ~ x.tail)
        yh.left <- as.numeric(trend$coefficients[1] + trend$coefficients[2] * end.left)
        if (as.numeric(table.left[1]) > threshold * yh.left) {
            out[1] <- end.left
        }
        i.tail <- 1:(L.right - 1)
        x.tail <- as.numeric(names(table.right)[i.tail])
        y.tail <- table.right[i.tail]
        trend <- lm(y.tail ~ x.tail)
        yh.right <- as.numeric(trend$coefficients[1] + trend$coefficients[2] * end.right)
        if (as.numeric(table.right[L.right]) > threshold * yh.right) {
            out[2] <- end.right
        }
    }

    return(out)
}


datstats <- function(x, ndig = NULL, na.rm = FALSE, invalid.rm = FALSE) 
{
    # Computes descriptive statistics of a vector, with handling of invalid data,
    # tests for distribution features ans sampling effects, and optional rounding of output.
    # 
    # Notes:
    #    1. If na.rm = FALSE:
    #           a) statistics that fail due to presence of NAs will return NA
    #               (assumes user was not expecting NAs).
    #    2. If na.rm = TRUE:
    #           a) NA will be counted as valid but ignored in computations
    #               (assumes user is actively managing NAs).
    #    3. If invalid.rm = FALSE:
    #           a) statistics that fail due to presence of invalid data will return NA
    #               (assumes user was not expecting invalid values).
    #    4. If invalid.rm = TRUE, invalid data are ignored (after being counted):
    #           Inf
    #           NULL
    #           NaN
    #           NA (if na.rm = FALSE).
    #
    # ISSUE: If x is integer, should results be truncated?
    #
    # TODO:
    #   1. Compute sumsqs, ...
    #   2. Create a "print" method, with default supression of "xv". (Need to make into an S3 class?)
    #
    
    if (!(typeof(x) %in% c("integer", "double"))) {
        print("descrstats(): Only numeric vectors are supported.")
        return(NA)
    }
    
    # Nb.  "n" is the original number of samples,
    #      "n.valid" is the number of valid samples (depends on na.rm),
    #      "n.used" is the number of samples used for computation (depends on 
    #           na.rm, invalid.rm).
    
    out <- list()
    
    out$n <- NROW(x)

    out$na.found <- sum(is.na(x))
    out$inf.found <- sum(is.infinite(x))
    out$null.found <- sum(is.null(x))
    out$nan.found <- sum(is.nan(x))
    
    if (na.rm)  #: keep NAs in xv but remove from x for computations
    {
        xv <- x
        x <- x[!is.na(x)]
        if (invalid.rm) {
            xv <- xv %>% is.finite %>% !is.null %>% !is.nan
            x <- x %>% is.finite %>% !is.null %>% !is.nan
        }
    } 
    else  #: remove NAs from xv but keep in x
    {
        xv <- x[!is.na(x)]
        if (invalid.rm) {
            xv <- xv %>% is.finite %>% !is.null %>% !is.nan
            x <- x %>% is.finite %>% !is.null %>% !is.nan
        }
    }
        
    out$n.used <- NROW(x)
    out$na.remain <- sum(is.na(x))
    out$n.valid <- NROW(xv)
    out$xv <- xv
    
    out$mean <- mean(x, na.rm = na.rm)
    out$median <- median(x, na.rm = na.rm)
    out$rms <- sqrt(mean(x ^ 2))
    out$min <- min(x, na.rm = na.rm)
    out$max <- max(x, na.rm = na.rm)
    out$skew <- skewness(x, na.rm = na.rm)  #: default estimator is b_1 (see pkg "e1071")
    out$kurt <- kurtosis(x, na.rm = na.rm)  #: default estimator is b_2 (see pkg "e1071")
    out$q95pc <- quantile(x, probs = c(0.95), names = FALSE)
    out$q10s <- quantile(x, probs = seq(0, 1, by = 0.1), names = TRUE)
    out$var <- var(x, na.rm = na.rm)
    out$sd <- sd(x, na.rm = na.rm)
    out$sem <- out$sd / out$n.used
    out$dip <- ifelse(out$na.remain > 0, NA, dip(x))
    out$clips <- censoring(x)
    
    if(!is.null(ndig)) {
        out$mean <- round(out$mean, ndig)
        out$median <- round(out$median, ndig)
        out$min <- round(out$min, ndig)
        out$max <- round(out$max, ndig)
        out$skew <- round(out$skew, ndig)
        out$kurt <- round(out$kurt, ndig)
        out$q95 <- round(out$q95, ndig)
        out$q10s <- round(out$q10s, ndig)
        out$var <- round(out$var, ndig)
        out$sd <- round(out$sd, ndig)
        out$sem <- round(out$sem, ndig)
        out$dip <- round(out$dip, ndig)
    }
    
    return(out)
}


ifsignif <- function(testStat, testOp = "<=", msg = NA, sigLevel = 0.05) 
{
    # Returns a string depending on result of comparing a test statistic to a
    # significance level.
    #
    # If msg is empty, the result will be "TRUE" or "FALSE";
    # else the result will be msg or empty string.
    #
    
    if (testOp == "<") {
        IsSignificant <- (testStat < sigLevel)
    } 
    if (testOp == "<=") {
        IsSignificant <- (testStat <= sigLevel)
    } 
    if (testOp == "=") {
        IsSignificant <- (testStat == sigLevel)
    }
    if (testOp == ">=") {
        IsSignificant <- (testStat >= sigLevel)
    }
    if (testOp == ">") {
        IsSignificant <- (testStat > sigLevel)
    }
    
    if (is.na(msg)) {
        outmsg <- ifelse(IsSignificant, "TRUE", "FALSE")
    } else {
        outmsg <- ifelse(IsSignificant, msg, "")
    }
    
    return(outmsg)
}


get.ndim.stats <- function(data) 
{
    # Computes number of rows in a vector or data frame.
    # (Copied from "signalysis.R")
    
    if (is.null(nrow(data))) {
        ndims <- 1
    } else {
        ndims <- length(data)
    }
    return(ndims)
}


modex <- function(x)  #: DEPRECATED!
{
    # Returns the mode of a distribution.
    # (Can be non-unique.)
    
    ux <- unique(x)
    # out <- ux[which.max(tabulate(match(x, ux)))]
    tab <- tabulate(match(x, ux))
    out <- ux[tab == max(tab)]
    
    return(out)
}

Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}


hist.discrete <- function(x, yax = "absolute")
{    
    # Plots a discrete histogram by counts or frequencies.
    
    if (!(yax %in% c("absolute", "relative"))) {
        print(paste0("hist.discrete(): undefined yax: ", hist.discrete))
    }
        
    if (yax == "absolute") {
        plot(table(x), type = "h", lwd = 5, ylab = "Abs Freq")
    } 
    if (yax == "relative") {
        plot(table(x) / length(x), type = "h", lwd = 5, ylab = "Rel Freq")
    }
}


randomseed <- function()
{
    # Generates a really random seed for set.seed().
    
    stock.code <- "GOOG"  
    # TODO: find a composite index that getQuote() doesn't return NA for !!!!!!!!!!!!!
    
    # quote <- getQuote(stock.code, src = "yahoo", what = yahooQF("Bid"))
    # s1 <- quote$Bid
    num <- getQuote(stock.code)$Last
    #print(quote)
    ndec <- decimalplaces(num)
    s1 <- num * 10 ^ ndec
    
    dt <- Sys.time()
    
    s2 <- gsub(':', '', as.character(dt), fixed=TRUE)
    s2 <- gsub('\\-', '', s2)
    s2 <- gsub(' ', '', s2)
    s2a <- as.numeric(gsub('\\-', '', strsplit(as.character(Sys.time()), ' ')[[1]][1])) 
    s2b <- as.numeric(gsub(':', '', strsplit(as.character(Sys.time()), ' ')[[1]][2])) 
    s2 <- s2a * s2b 
    s2 <- as.numeric(substr(as.character(s2), 1, 8))
    
    seed <- s1 * s2
    seed <- as.numeric(substr(as.character(seed), 1, 8))
    
    return(seed)
}


add.column <- function(x, y) 
{
    if (get.ndim(x) > 0) {
        y <- cbind(x, y)
    } 
    return(y)
}


summary.table <- function(x, stats, vector.names, stats.names = NULL)
{
    # Constructs a data table from summary statistics on the input vectors.
    
    df.in <- do.call(cbind, x)
    if (is.null(stats.names)) {
        names.in <- paste0(stats, ":")
    } else {
        names.in <- stats.names
    }
    
    y <- data.frame()
    if ("n" %in% stats) {
        y <- add.column(y, NROW(df.in))
    }
    if ("sum" %in% stats) {
        y <- add.column(y, sum(df.in))
    }
    if ("mean" %in% stats) {
        y <- add.column(y, mean(df.in))
    }
    if ("min" %in% stats) {
        y <- add.column(y, min(df.in))
    }
    if ("max" %in% stats) {
        y <- add.column(y, max(df.in))
    }
    print(y)
    
    return(y)   
}


binsort <- function(x, bins) 
{
    # Bins data as for a histogram and returns a list.
    #
    # Parameters:
    #
    #   bins        either a numeric vector of two or more unique cut points or a single number 
    #               (greater than or equal to 2) giving the number of intervals into which x is to be cut.
    #
    # TODO: More options for binning (e.g. levels based on cumulative probability or cumulative value).
    #
    
    cutb <- cut(x, breaks = bins)
    # print("cutb:")
    # print(cutb)
    
    if (NROW(bins) == 1) {
        nbins <- bins
    } else {
        nbins <- NROW(bins) - 1
    }
    
    xb <- list()
    for (i in 1:nbins) {
        # print(levels(cutb)[i])
        xb[[i]] <- x[cutb == levels(cutb)[i]]
    }

    return(xb)
}

# test:
#  binsort(runif(100, 0, 100), seq(0, 100, 10))


spiking <- function(x, method = "max", bin.breaks = NULL, bin.width = NULL, bin.window = 2, threshold = 2.0) 
{    
    # Tests a vector for spikes (i.e. local maxima in a binned distribution). Returns bin indices.
    #
    # Method:   Puts the samples into discrete bins, and employs one of several methods to 
    #           compare the count in each bin to a prediction based on the overall trend. 
    #           If a bin exceeds the predicted amount by a designated factor, spiking is true.
    #
    # Parameters:
    #
    #   method          how to define an outlier:
    #
    #                       "max"       for each bin, compare count of extreme value with counts of other 
    #                                   values within a local portion of the  distribution.
    #
    #                       "prob"      for each tail, compute probability of table[?] occurrences 
    #                                   of the extreme value -- TODO: NOT YET CODED !
    #
    #                       "trend"     for each tail, for a linear model to all bins, then compare the
    #                                   predicted and actual values. -- TODO: find a way to minimise bias 
    #                                   of spiky bins on trend!
    #
    #   threshold       multiplicative factor for comparison of tail-end to window (determines how 
    #                   much higher the tail-end peak must be than the highest peak in the window).
    #
    #   bin.width       implicit specification of number of bins.
    #
    #   bin.breaks      number of bins in distribution (scalar) or cut points for binning (vector).
    #
    #   bin.window      number of bins used in computation of local mean. (Should be an even number, for
    #                   symmetry about central point.)
    #
    # Notes:
    #
    #    1. For dynamic signals, sensitivity and selectivity can be strongly dependent on the arguments, 
    #       especially for small sample sizes (e.g. N=O(100) samples in each tail).
    #
    #    2. If threshold = 1, comparison will fail; therefore, is adjusted in code.
    #
    
    out <- integer()
    
    if (threshold == 1) { threshold <- 1.01 }  # round-off error (?) causes test to fail
    
    if (is.null(bin.breaks)) {
        if (is.null(bin.width)) {
            stop("spiking(): Either bin.breaks or resolution must be non-null.")
        }
        if (bin.width <= 0) {
            stop("spiking(): resolution must be positive real.")
        }
        numquantiles <- floor(1 / bin.width)
        bins <- cut2(x, g = numquantiles)
    } else {
        if (length(bin.breaks) > 1) {
            numquantiles <- length(bin.breaks)
            bins <- cut2(x, cuts = bin.breaks)
            stop("spikin(): cuts parameter does not work as expected!")
        } else {
            if (bin.breaks <= 0) {
                stop("spiking(): bin.breaks must be positive integral.")
            }
            numquantiles <- bin.breaks
            bins <- cut2(x, g = numquantiles)
        }
    }

    counts <- table(bins)
    nbins <- length(counts)
    
    max.bin.window <- (nbins - 1) / 2
    if (bin.window < 2 | bin.window > max.bin.window) {
        stop(paste0("statsfns::spiking(): bin.window = ", bin.window, " -- out of range (2 ~ ", max.bin.window, "."))
    }
    
    # browser()
    
    #E: trend <- lm(as.vector(counts) ~ (1:nbins))
    y <- as.vector(counts)
    x <- 1:nbins
    trend <- lm(y ~ x)
    yh <- predict(trend, x)
    
    if (method == "max") {
        # for each bin, compare count of extreme value with counts of other values in window of distribution ...

        for (i in 1:nbins) {
            window.left <- max(1, i - bin.window / 2)
            window.right <- min(i + bin.window / 2, nbins)
            
            if (count[i] > threshold * max(count[window.left:window.right])) {
                out <- rbind(out, i)
            }
        }
    } else if (method == "prob") {
        # for each window, compute probability of table[?] occurrences of the extreme value ...
        
        print(paste0("censoring(): Argument not yet supported: method = '", method, "' -- using 'trend'"))
        
    } else if (method == "trend") {
        # for each window, for a linear model to all bins except the extreme one, then compare the
        # predicted and actual values of the tail-end count ...
        
        # i.tail <- 2:(L.left)
        # x.tail <- as.numeric(names(table.left)[i.tail])
        # y.tail <- table.left[i.tail]
        # trend <- lm(y.tail ~ x.tail)
        # yh.left <- as.numeric(trend$coefficients[1] + trend$coefficients[2] * end.left)
        # if (as.numeric(table.left[1]) > threshold * yh.left) {
        #     out[1] <- end.left
        # }
        # i.tail <- 1:(L.right - 1)
        # x.tail <- as.numeric(names(table.right)[i.tail])
        # y.tail <- table.right[i.tail]
        # trend <- lm(y.tail ~ x.tail)
        # yh.right <- as.numeric(trend$coefficients[1] + trend$coefficients[2] * end.right)
        # if (as.numeric(table.right[L.right]) > threshold * yh.right) {
        #     out[2] <- end.right
        # }
    }
    
    return(out)
}


obfuscate <- function(in.ids, ndig = NULL, outfile = NULL)
{    
    # Returns a vector of anonymised identifiers randomly mapped to the input vector.
    # 
    # Optionally saves the lookup table to a file for deanonymisation.
    
    lookup <- obfuscator(in.ids = in.ids, ndig = ndig)
    stop("UNFINISHED! Use obfuscator() instead.")
    

}


obfuscator <- function(in.ids, ndig = NULL, out.names = NULL, out.type = "data.frame")
{    
    # Generates a lookup table of input identifiers and randomly mapped output identifiers appropriate 
    # to the input range with specified number of digits.
    #
    # Output is a out.type = data.frame by default. If outtype = "data.table" specified, the key will 
    # set automatically.
    #
    # TODO
    #    1. Output range should be not necessaily be spanned by output; skewness could be used to infer 
    #       original range. May want to conceal magnitude of underlying data set in subset. Allow user to: 
    #         a. specify "sparse" to exceed range of input data 
    #         b. option to specify range 
    #       

    # set.seed(1)  #: test only
    
    #D: print(paste0("obfuscator(): str(in.ids) = ", str(in.ids)))
    
    numids <- NROW(in.ids)

    # if(is.all.integer(in.ids)) {  #:< NOT NEEDED !
        # create vector of substitute patient codes ...
        
    if (is.null(ndig)) {
        # renumber to homogeneous scale:
        ndig <- ceiling(log10(numids))
        print(paste0("obfuscator(): ndig not passed; defaults to ", ndig))
    } else {
        ndig0 <- ceiling(log10(numids))
        if (10 ^ (ndig0 + 1) < numids) {
            print(paste0("obfuscator(): Specified number of digits too small for input count; increasing to ", ndig0))
            ndig <- ndig0
        } else {
            print(paste0("obfuscator(): ndig passed as ", ndig))
        }
    }
    # out.ids <- (1:numids) + 10 ^ (ndig + 1)  # this does not ensure constant number of digits, and is too many digits!
    out.ids <- 10 ^ ndig - (1:numids)    # this ensures constant number of digits, but omits those from teh lower end of the range
    
    # randomise the output sequence and truncate to same length as input:
    #X: randids <- factor(sample(out.ids, length(in.ids), replace = FALSE))
    randids <- sample(out.ids, length(in.ids), replace = FALSE) 
    
    # create lookup table:
    out.map <- data.frame(inID = in.ids, outID = randids, stringsAsFactors = FALSE)

    #D: print(paste0("obfuscator(): str(out.map) = ", str(out.map)))
    
    # } else {
    #     print("obfuscator(): Not all vector elements are integral; alphanumeric IDs not yet supported!")
    #     out.map <- NA
    # }
    
    if (!is.null(out.names)) {
        names(out.map) <- out.names
    }
    
    if (out.type == "data.table") {
        out.map <- data.table(out.map, key = names(out.map)[1])
    }
    
    return(out.map)
}


getTail <- function(x, column = NULL, thresh = NULL, frac = NULL, tail = 'right', boundary = TRUE) {
    
    # Returns the subset of a distribution whose values exceed a threshold 
    # (or the subset of a data frame based on the above).
    #
    # Threshold can be passed as an absolute number or a fraction of the maximum value
    # of the distribution.
    
    if (class(x) == 'data.frame') {
        if (is.null(column)) {
            stop('getTail(): No column specified.')
        } else {
            df <- x
            if (class(column) == 'numeric') {
                icol <- column
            } else {
                icol <- which(names(df) == column)
            }
            xc <- df[, icol]
            #X: rm(df)
        }
    } else {
        xc <- x
    }
    
    if (is.null(thresh)) {
        if (is.null(frac)) {
            stop('getTail(): thresh and frac cannot both be NULL.')
        } else {
            fraction <- as.numeric(frac)
            if (tail == 'right') {
                x.t <- fraction * (max(xc, na.rm = TRUE) - min(xc, na.rm = TRUE)) + min(xc, na.rm = TRUE)
            } else {
                x.t <- (1 - fraction) * (max(xc, na.rm = TRUE) - min(xc, na.rm = TRUE)) + min(xc, na.rm = TRUE)
            }
        }
    } else {
        threshold <- as.numeric(thresh)
        quantile.name <- names(frac)[1]  # not yet used
        x.t <- threshold
    }
    
    if (tail == 'right') {
        if (boundary) {
            xs <- xc[xc >= x.t]
        } else {
            xs <- xc[xc > x.t]
        }
    }
    
    if (tail == 'left') {
        if (boundary) {
            xs <- xc[xc <= x.t]
        } else {
            xs <- xc[xc < x.t]
        }
    }

    # if (is.null(column)) {
    if (class(x) == 'data.frame') {
        dfs <- df[df[, icol] %in% xs, ]
        return(dfs)
    } else {
        return(xs)
    }
}


extractMatrix <- function (x) {
    
    # Extracts a matrix from a data frame, ignoring non-numeric columns.
    
    numeric.cols <- sapply(x, is.numeric)
    return(x[, numeric.cols])
}


compare <- function(m1, m2, name = 'sample', s1 = NULL, s2 = NULL, n1 = NULL, n2 = NULL, verbose = TRUE) {
    
    # Compares two values (absolute & percent difference, significance).
    
    # Can only compute significance if sdev (s1, s2) & nsamples (n1, n2) passed (assuming normal distribution).
    
    if (m1 != 0) {
        diff <- percentdiff(m1, m2, ndec = 0, string = FALSE)
        csign <- ifelse(diff >= 0, '+', '-')
        out <- paste0(csign, abs(diff), '%')
        
        if (verbose) {
            cat(paste0(name, ':'), '1 =', m1, ', 2 =', m2, 
                ', diff =', out, '\n')
        }
    } else {
        out <- 'NA'
    }
    
    return(out)
}


confusion <- function(actual, predicted)  # , na.rm = FALSE
{
    # Computes stats from confusion matrix.
    
    n <- NROW(actual)
    P <- sum(actual == TRUE)
    N <- sum(actual == FALSE)
    pP <- sum(predicted == TRUE)    #: predicted positives
    pN <- sum(predicted == FALSE)   #: predicted negatives
    TP <- sum(actual == TRUE & predicted == TRUE)
    TN <- sum(actual == FALSE & predicted == FALSE)
    FP <- sum(actual == FALSE & predicted == TRUE)
    FN <- sum(actual == TRUE & predicted == FALSE)
    
    a <- list()
    a$PR <- P / n
    a$NR <- N / n
    a$TPR <- TP / P
    a$TNR <- TN / N
    a$FPR <- FP / P
    a$FNR <- FN / N
    a$pPR <- pP / n
    a$pNR <- pN / n
    a$sensitivity <- TP / P
    a$recall <- a$sensitivity
    a$specificity <- TN / N
    a$precision <- a$specificity
    a$PPV <- TP / (TP + FP)
    a$NPV <- TN / (FN + TN)
    a$accuracy <- (TP + TN) / n
    
    return(a)
}


