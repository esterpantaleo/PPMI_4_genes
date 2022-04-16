#author:	Ester Pantaleo
#this code is free and can be used, adapted, copied or published without permission	provided that the associated publication is	cited:
#E. Pantaleo et al. A Machine Learning approach to Parkinson’s disease blood transcriptomics, Genes 2022
#this is part of the code used therein to run the final xgboost routine, with the selected set of features (genes)

args = commandArgs(trailingOnly=TRUE)

#get the first argument
my_seed = args[1]
set.seed(my_seed)

library(MLeval)
library(caret)
library(dplyr)
library(xgboost)
load("data.filtered.RData")
load("y.filtered.RData")
load("gene_tab.RData")
load("ensembl_version_ids.RData")

s = which(colnames(data) %in% ensembl_version_ids)
X = data[, s]

foldIndex <- createFolds(factor(y), k = 10, list = FALSE)
save(foldIndex, file = paste0("./results/", my_seed, "/foldIndex.RData"))
performances = c()
for (j in 1:10){
    print(paste("performing run", j, "of cross validation..."))

    train <- data.frame(y = as.factor(y[foldIndex != j]), X = X[foldIndex != j, ])
    train_control = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary)
    bst <- train(y ~ .,
                     data = train,
                     method = "xgbTree",
                     metric = "ROC",
                     trControl = train_control)
    test <- data.frame(y = as.factor(y[foldIndex == j]), X = X[foldIndex == j, ])
    prediction <- predict(bst, newdata = test, type = "prob")
    test$y_pred = as.factor(colnames(prediction)[max.col(prediction, ties.method = "first")])
    
    rocauc = evalm(data.frame(prediction, as.factor(test$y)))$stdres$Group1$Score[13]
    cm = confusionMatrix(data = test$y_pred, reference = test$y, positive = "PD")
    print(cm)
    specificity <- cm$byClass['Specificity']
    sensitivity <- cm$byClass['Sensitivity']
    balanced_accuracy <- cm$byClass['Balanced Accuracy']
    F1 <- cm$byClass['F1']
    accuracy <- cm$overall['Accuracy']
    performances = cbind(performances, c(rocauc, sensitivity, specificity, F1, balanced_accuracy, accuracy))
    imp = varImp(bst)$importance

    save(prediction, file = paste0("./results/", my_seed, "/prediction_", j, ".RData"))
    save(cm, file = paste0("./results/", my_seed, "/cm_", j, ".RData"))
    save(imp, file = paste0("./results/", my_seed, "/imp_", j, ".RData"))
}