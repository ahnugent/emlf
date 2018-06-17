# File:       cleanly_ex.R
# Authors:    Allen H Nugent, Dec'15+
# Last edit:  2018-06-14
# Last test:  2018-06-14
# Purpose:    Data cleaning functions.
#
# NOTE: 'cleanly_ex.R' is a redacted subset of 'cleanly.R': not all functions are available!
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# Contents:
#
#  bool.to.char     Returns a logical character/string from a boolean.
#
#  bool.to.logical  Returns a logical from a boolean integer.   
#
#  int.to.logical   Returns a logical from a boolean.
#
#  logical.to.bool  Returns a boolean integer from a logical.
#
#  filldates        Returns a data.frame with missing dates filled in.
#
#  format.path      Converts a Windows path to the format used in R.
#
#  get.col.num      Returns column number pertaining to the name provided.
#
#  get.col.nums     Returns a vector of column number pertaining to the names provided.
#  
#  is.all.integer   Returns TRUE if all members of the input vector are integral.
#
#  is.wholenumber   Returns TRUE if argument is a whole number.
#
#  lookup           Returns a vector of lookup values from lookupTable using keyVector as indices.
#
#  na.replace       Replaces NA within a vector with specified value.
#
#  nunique          Returns count of unique elements.
#
#  set.factorsp     Defines a factor based on a set of levels and an arbitrary sequence,
#                   and applies it to the input vector.
#
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



is.wholenumber <- function(x, tol = .Machine$double.eps ^ 0.5)  
{
    return(abs(x - round(x)) < tol)
}


is.all.integer <- function(x)
{   
    # Returns TRUE if all members of the input vector are integral.
    
    if ((sum(sapply(x, is.numeric)) != NROW(x))) {
        out <- FALSE
    } else {
        out <- (sum(sapply(x, is.wholenumber)) == NROW(x))
    }
    
    return(out)
}


format.path <- function(inpath) {

    # Converts a Windows path to the format used in R.
    
    outpath <- sub("\\\\","/", inpath)
    return(outpath)
}


lookup <- function(keyVector, lookupTable) 
{
    # Returns a vector of lookup values from lookupTable using keyVector as indices.
    #
    # Assumptions:
    #
    #   1. lookupTable is a the data frame with:
    #           1st column: name = "code"; contents = key codes  
    #           2nd column: name = "category"; contents = values ot be looked up, returned.
    #
    #   2. If NAs are to be handled explicitly, lookupTable should map replacement value to an NA level in the "code" column.
    #       ##> BUGGY!					  
    #
    
    # get lookup values:
    outVector <- lookupTable$category[keyVector]
    
    # handle empty strings, missing values:
    # (Nb. empty strings should already have been converted to NA by the inital lookup step, above)
    # BUGGY: THROWS ERROR: outVector[is.na(outVector) | outVector == ""] <- lookupTable$category[is.na(lookupTable$code)]
	# NEW ...
    outVector[outVector == ""] <- NA
    if (sum(is.na(lookupTable$code)) > 0) {
        na_alias <- lookupTable$category[is.na(lookupTable$code)]
        outVector[is.na(outVector)] <- na_alias
    }
    
    return(outVector)
}


get.col.num <- function(df, col.name)
{
    # Returns column number pertaining to the name provided.
    return(which(colnames(df) == col.name))
}

get.col.nums <- function(df, col.names)
{    
    # Returns a vector of column number pertaining to the names provided.
    #X: return(sapply(df, get.col.num, col.names)) >> output includes 0 & 1 results
    #X: return(vapply(df, get.col.num, col.names))
    #X: col.nums <- get.col.num(col.names)
    col.nums <- integer(length(col.names))
    for (i in 1:length(col.names))
    {
        col.nums[i] <- get.col.num(df, col.names[i])
    }
    return(col.nums)
}


set.factorsp <- function(x, x.levels.str, x.factors.str)
{
    # Defines a factor based on a set of levels and an arbitrary sequence,
    # and applies it to the input vector.
    # Both the levels and factors are supplied as newline-separated strings.
    
    # parse the input strings into vectors:
    x.levels <- unlist(strsplit(x.levels.str, '\n'))
    x.factors <- as.integer(unlist(strsplit(x.factors.str, '\n')))
    
    # reorder levels according to order of factors (in case non-monotonic):
    x.levels <- x.levels[order(x.factors)]
    
    # convert to factor:
    f <- x.levels[x]
    #?: x <- factor(x.levels[x], x.levels, ordered = TRUE)
    x <- factor(x.levels[x], x.levels)
 
    return(x)   
}


set.levels <- function(x, x.levels.str, x.factors.str, ordered = FALSE)
{
    # Replaces a factor-like variable based on a set of levels and an arbitrary sequence,
    # and applies it to the input vector.
    # Both the levels and factors are supplied as newline-separated strings.
    
    # parse the input strings into vectors:
    x.levels <- unlist(strsplit(x.levels.str, '\n'))
    x.factors <- as.integer(unlist(strsplit(x.factors.str, '\n')))
    
    # reorder levels according to order of factors (in case non-monotonic):
    x.levels <- x.levels[order(x.factors)]

    # apply to input vector:
    f <- x.levels[x]
    
    if (ordered) {
        f <- factor(f, x.levels, ordered = TRUE)
    } else {
        f <- factor(f, x.levels)
    }

    return(f)
}


na.replace <- function (x, withval = 0) 
{
    # Replaces NA within a vector with specified value.
    # (Overcomes issue with replacing elements of a data.frame)
    
    x[is.na(x)] <- withval
    return(x)
}


filldates <- function(all_dates, d, col.date = 1, col.data = 2)
{
    # Returns a data.frame with missing dates filled in (as per all_dates).
    # Replaces NAs in data column with 0.
    
    d <- merge(all_dates, d, by = col.date, all.x = TRUE)
    d[, col.data] <- na.replace(d[, col.data], 0)
    return(d)
}


bool.to.char <- function(b, outform = 'N/Y') {
    
    # Returns a logical character from a boolean or logical.
    
    if (outform == 'N/Y') {
        strF <- 'N'
        strT <- 'Y'
    }
    if (outform == 'F/T') {
        strF <- 'F'
        strT <- 'T'
    }
    
    return(ifelse(b, strT, strF))
}


logical.to.bool <- function(x) {

    # Returns a boolean integer from a logical.     
    
    return(as.integer(x))
}


int.to.logical <- function(x) {
    
    # Returns a logical from an integer.   
    
    # x == 0    returns FALSE
    # x != 0    returns TRUE
    
    return(as.logical(x))
}


bool.to.logical <- function(x, strict = TRUE) {
    
    # Returns a logical from a boolean integer, with (default) strict domain testing.  
    
    # x == 0    returns FALSE
    # x == 1    returns TRUE
    # else      returns NA
    
    return(ifelse(x == 1, TRUE, ifelse(x == 0, FALSE, NA)))
}


nunique <- function(x, do.trim = TRUE) {
    
    # Returns count of unique elements.
    
    if (do.trim) { 
        x <- trim(x)
    }
    return(length(unique(x)))
}


