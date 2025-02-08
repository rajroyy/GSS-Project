gss = read.csv("/Users/rajroy/Documents/Statistics/Assignment 2/GSS_HappyData.csv")

gss2 = gss[, c( "VHAPPY", "FamIncomeAdj", "SATFIN", "FINRELA")]

#View(gss2)
summary(gss2)

#replace FamIncomeAdj NA's w/ averages
avg.FamIncomeAdj <- mean(gss2$FamIncomeAdj, na.rm = TRUE)
gss2$FamIncomeAdj[is.na(gss2$FamIncomeAdj)==TRUE] <- avg.FamIncomeAdj
summary(gss2)

#omit rest of NAs
gss2 = na.omit(gss2)
summary(gss2)

#check relationships b/w predictors with corr matrix
variables <- gss2[, c("FamIncomeAdj", "SATFIN", "FINRELA")]

cor(variables)

#### LOG REGRESSION CHECKS/ASSUMPTION CHECKS ####

#FamIncomeAdj

gss2$FamIncomeAdj_Binned = cut(gss2$FamIncomeAdj, breaks = 20, include.lowest = TRUE)

sample_byFamIncomeAdj = table(gss2$FamIncomeAdj_Binned)
sample_byFamIncomeAdj

VHAPPY_byFamIncome = table(gss2$FamIncomeAdj_Binned[gss2$VHAPPY == 1])
VHAPPY_byFamIncome

data.famincomeadj = data.frame(FamIncomeAdj_Binned = names(sample_byFamIncomeAdj),
                               counts = as.numeric(sample_byFamIncomeAdj),
                               happy = as.numeric(VHAPPY_byFamIncome))
data.famincomeadj

data.famincomeadj$prop = data.famincomeadj$happy/data.famincomeadj$count

barplot(data.famincomeadj$prop ~ data.famincomeadj$FamIncomeAdj_Binned, las = 2)

data.famincomeadj$odds = data.famincomeadj$prop/(1-data.famincomeadj$prop)
data.famincomeadj$logodds = log(data.famincomeadj$odds)

data.famincomeadj$FamIncomeAdj_Binned_mean = tapply(gss2$FamIncomeAdj, gss2$FamIncomeAdj_Binned, mean, na.rm = TRUE)

plot(data.famincomeadj$FamIncomeAdj_Binned_mean, data.famincomeadj$logodds)
abline(lm(data=data.famincomeadj, logodds ~FamIncomeAdj_Binned_mean), col = "red")

#SATFIN

sample_bySATFIN = table(gss2$SATFIN)
sample_bySATFIN

VHAPPY_bySATFIN = table(gss2$SATFIN[gss2$VHAPPY == 1])
VHAPPY_bySATFIN

data.satfin = data.frame(SATFIN = as.numeric(names(sample_bySATFIN)),
                         counts = as.numeric(sample_bySATFIN),
                         happy = as.numeric(VHAPPY_bySATFIN))
data.satfin

data.satfin$prop = data.satfin$happy / data.satfin$counts
data.satfin$odds = data.satfin$prop / (1 - data.satfin$prop)
data.satfin$logodds = log(data.satfin$odds)

barplot(data.satfin$prop ~ data.satfin$SATFIN, las = 2)

plot(data.satfin$SATFIN, data.satfin$logodds, xlab = "SATFIN", 
     ylab = "Log(odds)")
abline(lm(data = data.satfin, logodds ~SATFIN), col = "red")

#FINRELA

sample_byFINRELA = table(gss2$FINRELA)
sample_byFINRELA

VHAPPY_byFINRELA = table(gss2$FINRELA[gss2$VHAPPY == 1])
VHAPPY_byFINRELA

data.finrela = data.frame(FINRELA = as.numeric(names(sample_byFINRELA)),
                          counts = as.numeric(sample_byFINRELA),
                          happy = as.numeric(VHAPPY_byFINRELA))
data.finrela

# Calculate proportions, odds, and log odds

data.finrela$prop = data.finrela$happy  / data.finrela$counts 
data.finrela$odds = data.finrela$prop / (1 - data.finrela$prop)
data.finrela$logodds = log(data.finrela$odds)

barplot(data.finrela$prop ~ data.finrela$FINRELA, las = 2)

plot(data.finrela$FINRELA, data.finrela$logodds, xlab = "FINRELA", 
     ylab = "Log(odds)")
abline(lm(data = data.finrela, logodds ~FINRELA), col = "red")

#### LOGISTIC REGRESSION ####

gl1 = glm(data = gss2, VHAPPY ~ FamIncomeAdj + SATFIN + FINRELA, family = binomial)
summary(gl1)

#### MODEL METRICS ####

gss2$predicted.prob = predict(gl1, type = "response")
gss2$predicted.class = ifelse(gss2$predicted.prob >= 0.5, 1, 0)

confusion_matrix = table(predicted = gss2$predicted.class,
                         actual = gss2$VHAPPY)
confusion_matrix

#accuracy: measures the percentage of true predictions as a whole: (0.70)
accuracy = sum(diag(confusion_matrix))/sum(confusion_matrix)

#precision: this calculates the percentage of predicted positives: 
precision = confusion_matrix[2,2]/sum(confusion_matrix[2,])
#from those predictions that were predicted as true, only 49% were actually true

#sensitivity (recall) = the percentage of true positives in terms of the total positives (TP/TP+FN)
sensitivity = confusion_matrix[2,2]/sum(confusion_matrix[,2])

#specificity: (TN/TN + FP)
specificity = confusion_matrix[1,1]/sum(confusion_matrix[,1])

#F1 score: 2*precision*sensitivity/(precision + sensitivity)
f1 = 2*precision*sensitivity/(precision + sensitivity)
f1 #model is not doing well with this particular threshold - 0.10319

###ROC and AUC
#install.packages("pROC")
library(pROC)
roc.obj = roc(gss2$VHAPPY, gss2$predicted.prob)
plot(roc.obj, col = "red")
auc(roc.obj)


#### LASSO alternative model selection ####

#install.packages("glmet")
library(glmnet)

var.list <- c("FamIncomeAdj", "SATFIN", "FINRELA")

lasso1.preds <- gss2[, var.list] #Predictor variables
DV <- gss2$VHAPPY             #Response variable

cv.lasso1 <- cv.glmnet(y = as.matrix(DV), 
                      x = as.matrix(lasso1.preds),
                      family = "binomial", ## LASSO on log regression = "binomial"
                      na.action = NULL,
                      type.measure = "auc")

#Extracting coefficients at optimal lambda min val
lasso_coefficients = coef(cv.lasso1, cv.lasso1$lambda.min)

print(lasso_coefficients)

plot(cv.lasso1)

