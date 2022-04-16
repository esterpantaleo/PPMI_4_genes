#author: Ester Pantaleo
#this code is free and can be used, adapted, copied or published without permission provided that the associated publication is cited:
#E. Pantaleo et al. A Machine Learning approach to Parkinson’s disease blood transcriptomics, Genes 2022
#this is part of the code used therein to run the RF routine, as explained in the pseudocode
library(doParallel)
library(caret)
library(dplyr)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)
args <- commandArgs(trailingOnly = TRUE)

#get first and second argument
#h = takes values btw 1 and 10
#kk = takes values btw 1 and 100
h = as.integer(args[1])
kk = as.integer(args[2])

#print log
sink(paste0("log_", kk, "/log_", h, ".log"))

print(paste0("cv run ", h, " using file cv.", h, ".RData in folder ", kk))

setwd(paste0("output_", kk, "/"))

print("loading data")
load(paste0("cv.", h, ".RData"))

data = cbind(cv$train$data, cv$train$gen)
colnames(data)[ncol(data)] <- "gen"
y = cv$train$y

for (i in 1:100){
    print(paste("seed", i))
    set.seed(i)

    # Folds are created on the basis of target variable
    foldIndex <- createFolds(factor(y), k = 5, list = FALSE)
    print("saving foldIndex")
    save(foldIndex, file = paste0("RFfoldIndex.", h, ".", i, ".RData"))
    output = list("feat_importance" = matrix(rep(-1, dim(data)[2]), ncol = 1),
           "binary_importance" = matrix(rep(0, dim(data)[2]), ncol = 1))
    j = 1
    train <- list(data = data[foldIndex != j, ], y = as.factor(y[foldIndex != j]))
    fit_control = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary, sampling = "down", method = "cv", number = 5)
    print(paste("running random forest", h, i))
    rf_fit <- train(train$data, train$y, method = 'rf', ntree = 1000, metric = "ROC", trControl = fit_control, do.trace = 25)

    print("saving outliers only")
    importances = varImp(rf_fit)$importance[, "Overall"]
    median_importance = median(importances)
    IQR = quantile(importances, 0.75) - quantile(importances, 0.25)
    thr = median_importance + 1.5 * IQR
    inds = which(importances > thr)
    output$feat_importance[, 1] <- importances
    output$binary_importance[inds, 1] = 1
    print("saving results")
    write.table(output$binary_importance, sep = ",", row.names = FALSE, quote = FALSE, file = paste0("binary_importances.", h, ".", i, ".csv"))
    write.table(output$feat_importance, sep = ",", row.names = FALSE, quote = FALSE, file = paste0("feat_importances.", h, ".", i, ".csv"))
}
stopCluster(cl)
print("job completed")