library(tidyverse)
library(caret)
library(MASS)
library(class)
library(tree)
library(randomForest)
library(gbm)
options(scipen = 99)
p <- read_csv("pubDataRepo.csv")
# Data set available for download at https://projects.huffingtonpost.com/projects/ncaa/reporters-note
forlm <- p[,c(13:21, 24)]

# Question 1

## How well can a linear regression project an athletic program’s spending by examining figures related the program’s firmographics and income?

### Linear Regression

# Ensure results are reproducible
set.seed(90)

#Create a training set and test set using a 70-30 split of forlm
inlmTrain <- createDataPartition(forlm$inflation_adjusted_athletic_expenses, p = 0.7, list = FALSE)
forlmTrain <- forlm[inlmTrain,]
forlmTest <- forlm[-inlmTrain,]

#Creating the linear model
lmmodel <- lm(inflation_adjusted_athletic_expenses ~ ., data = forlmTrain)
summary(lmmodel)

#Run test set thru model
pred <- predict(lmmodel, forlmTest)

#Calculate residuals
error <- pred - forlmTest[["inflation_adjusted_athletic_expenses"]]

#Calculate root mean square error from residuals between predicted and actual values
sqrt(mean(error^2))

#Plot the residuals to look for problems
par(mfrow = c(2,2))
plot(lmmodel)

#Cross valdation
set.seed(90)
lmmodelcv <- train(
  inflation_adjusted_athletic_expenses ~ ., 
  data = forlm,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 10
))
summary(lmmodelcv)
lmmodelcv

# Question 2, Part A

## Can a logistic regression model correctly guess whether an athletic program belongs to the “power five" based on its income and spending figures?

### Logistic Regression

#Creating dataset for logistic regression
forlogit <- p[,c(6, 13:21, 24)]
forlogit$isPowerFive <- as.factor(forlogit$isPowerFive)

#Splitting into training and test sets
set.seed(90)
inlogitTrain <- createDataPartition(forlogit$isPowerFive, p = 0.7, list = FALSE)
forlogitTrain <- forlogit[inlogitTrain,]
forlogitTest <- forlogit[-inlogitTrain,]

#Creating model
logitmodel <- glm(isPowerFive ~ ., data = forlogitTrain, family = binomial)
summary(logitmodel)

#Run test data thru model
logitprobs <- predict(logitmodel, forlogitTest, type = "response")

#Set up values under 0.5 to "No" and values above 0.5 to "Yes"
logitpred = rep("No", 301)
logitpred[logitprobs > 0.50] = "Yes"

#Confusion matrix to check the model's accuracy
table(logitpred, forlogitTest$isPowerFive)
mean(logitpred == forlogitTest$isPowerFive)

#Cross validation
set.seed(90)
logitmodelcv <- train(
  isPowerFive ~ ., 
  data = forlogit,
  method = "glm",
  family = binomial,
  trControl = trainControl(
    method = "cv", 
    number = 10
))
summary(logitmodelcv)

#Run the test data thru the cross-validated model
logitcvprobs <- predict(logitmodelcv, forlogitTest)

#Test cv results
confusionMatrix(logitcvprobs, forlogitTest$isPowerFive)

### Linear Discriminant Analysis

#creating the lda model (we can reuse the logit splits here)
set.seed(90)
ldamodel = lda(isPowerFive ~ ., data = forlogitTrain)
ldamodel
plot(ldamodel)

#Run test set thru model
ldapred = predict(ldamodel, forlogitTest)
ldapredclass = ldapred$class

#Test results
table(ldapredclass, forlogitTest$isPowerFive)

#Calculate accuracy
mean(ldapredclass == forlogitTest$isPowerFive)

#Cross validation
set.seed(90)
ldamodelcv <- train(
  isPowerFive ~ ., 
  data = forlogit,
  method = "lda",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated cv model
ldacvprobs <- predict(ldamodelcv, forlogitTest)
confusionMatrix(ldacvprobs, forlogitTest$isPowerFive)

### Quadratic Discriminant Analysis

#creating the qda model (we can reuse the logit splits here)
set.seed(90)
qdamodel = qda(isPowerFive ~ ., data = forlogitTrain)
qdamodel

#Run test set thru model
qdapred = predict(qdamodel, forlogitTest)
qdapredclass = qdapred$class

#Test results
table(qdapredclass, forlogitTest$isPowerFive)

#Calculate accuracy
mean(qdapredclass == forlogitTest$isPowerFive)

#Cross validation
set.seed(90)
qdamodelcv <- train(
  isPowerFive ~ ., 
  data = forlogit,
  method = "qda",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated cv model
qdacvprobs <- predict(qdamodelcv, forlogitTest)
confusionMatrix(qdacvprobs, forlogitTest$isPowerFive)

### K-Nearest Neighbor Analysis

#### With k=1, k=3 and k=5

#knn requires matricies, so cbind is used to creat them for train and test
set.seed(90)
knnTrain <- cbind(forlogit$isPowerFive, forlogit$inflation_adjusted_ticket_sales, forlogit$inflation_adjusted_student_fees, forlogit$inflation_adjusted_direct_state_govt_support, forlogit$inflation_adjusted_direct_institutional_support, forlogit$inflation_adjusted_indirect_facil_admin_support, forlogit$inflation_adjusted_ncaa_distributions, forlogit$inflation_adjusted_royalties, forlogit$inflation_adjusted_tv_revenue, forlogit$inflation_adjusted_endowments, forlogit$inflation_adjusted_athletic_expenses)[inlogitTrain,]

knnTest <- cbind(forlogit$isPowerFive, forlogit$inflation_adjusted_ticket_sales, forlogit$inflation_adjusted_student_fees, forlogit$inflation_adjusted_direct_state_govt_support, forlogit$inflation_adjusted_direct_institutional_support, forlogit$inflation_adjusted_indirect_facil_admin_support, forlogit$inflation_adjusted_ncaa_distributions, forlogit$inflation_adjusted_royalties, forlogit$inflation_adjusted_tv_revenue, forlogit$inflation_adjusted_endowments, forlogit$inflation_adjusted_athletic_expenses)[-inlogitTrain,]

#attaching simplifies coding for coming step
attach(forlogit)

#knn requires a marker for what is its training data
knnTestMarker = isPowerFive[inlogitTrain]

#attaching simplifies coding for coming step
detach(forlogit)

#knn model with k = 1
set.seed(90)
knnmodel <- knn(knnTrain, knnTest, knnTestMarker, k = 1)
attach(forlogitTest)
table(knnmodel, isPowerFive)
mean(knnmodel == isPowerFive)

#knn model with k = 3
set.seed(90)
knn3model <- knn(knnTrain, knnTest, knnTestMarker, k = 3)
table(knn3model, isPowerFive)
mean(knn3model == isPowerFive)

#knn model with k = 5
set.seed(90)
knn5model <- knn(knnTrain, knnTest, knnTestMarker, k = 5)
table(knn5model, isPowerFive)
mean(knn5model == isPowerFive)

#cross validation
set.seed(90)
knnmodelcv <- train(
  isPowerFive ~ ., 
  data = forlogit,
  method = "knn",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated model
knncvprobs <- predict(knnmodelcv, forlogitTest)
confusionMatrix(knncvprobs, forlogitTest$isPowerFive)

# Question 2, Part B

## Can a logistic regression model correctly guess whether an athletic program belongs to the “power five,” “group of five,” or neither based on its income and spending figures?

### Linear Discriminant Analysis

forbonus <- p[,c(5, 13:20, 23, 24)]
forbonus$grouping <- as.factor(forbonus$grouping)

#Splitting into training and test sets
set.seed(90)
inbonusTrain <- createDataPartition(forbonus$grouping, p = 0.7, list = FALSE)
forbonusTrain <- forbonus[inbonusTrain,]
forbonusTest <- forbonus[-inbonusTrain,]

#creating the lda model (we can reuse the logit splits here)
set.seed(90)
ldabonusmodel = lda(grouping ~ ., data = forbonusTrain)
ldabonusmodel

#Run test set thru model
ldabonuspred = predict(ldabonusmodel, forbonusTest)
ldabonuspredclass = ldabonuspred$class

#Test results
table(ldabonuspredclass, forbonusTest$grouping)

#Calculate accuracy
mean(ldabonuspredclass == forbonusTest$grouping)

#Cross validation
set.seed(90)
ldabonusmodelcv <- train(
  grouping ~ ., 
  data = forbonus,
  method = "lda",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated cv model
ldabonuscvprobs <- predict(ldabonusmodelcv, forbonusTest)
confusionMatrix(ldabonuscvprobs, forbonusTest$grouping)

### Quadratic Discriminant Analysis

#creating the qda model (reusing lda splits)
set.seed(90)
qdabonusmodel = qda(grouping ~ ., data = forbonusTrain)
qdabonusmodel

#Run test set thru model
qdabonuspred = predict(qdabonusmodel, forbonusTest)
qdabonuspredclass = qdabonuspred$class

#Test results
table(qdabonuspredclass, forbonusTest$grouping)

#Calculate accuracy
mean(qdabonuspredclass == forbonusTest$grouping)

#Cross validation
set.seed(90)
qdabonusmodelcv <- train(
  grouping ~ ., 
  data = forbonus,
  method = "qda",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated cv model
qdabonuscvprobs <- predict(qdabonusmodelcv, forbonusTest)
confusionMatrix(qdabonuscvprobs, forbonusTest$grouping)

### K-Nearest Neighbor Analysis

#### With k=1, k=3 and k=5

set.seed(90)
knnbonusTrain <- cbind(forbonus$grouping, forbonus$inflation_adjusted_ticket_sales, forbonus$inflation_adjusted_student_fees, forbonus$inflation_adjusted_direct_state_govt_support, forbonus$inflation_adjusted_direct_institutional_support, forbonus$inflation_adjusted_indirect_facil_admin_support, forbonus$inflation_adjusted_ncaa_distributions, forbonus$inflation_adjusted_royalties, forbonus$inflation_adjusted_tv_revenue, forbonus$inflation_adjusted_other_revenues, forbonus$inflation_adjusted_athletic_expenses)[inbonusTrain,]

knnbonusTest <- cbind(forbonus$grouping, forbonus$inflation_adjusted_ticket_sales, forbonus$inflation_adjusted_student_fees, forbonus$inflation_adjusted_direct_state_govt_support, forbonus$inflation_adjusted_direct_institutional_support, forbonus$inflation_adjusted_indirect_facil_admin_support, forbonus$inflation_adjusted_ncaa_distributions, forbonus$inflation_adjusted_royalties, forbonus$inflation_adjusted_tv_revenue, forbonus$inflation_adjusted_other_revenues, forbonus$inflation_adjusted_athletic_expenses)[-inbonusTrain,]

#attaching simplifies coding for coming step
attach(forbonus)

#knn requires a marker for what is its training data
knnbonusTestMarker = grouping[inbonusTrain]

#attaching simplifies coding for coming step
detach(forbonus)

#knn model with k = 1
set.seed(90)
knnbonusmodel <- knn(knnbonusTrain, knnbonusTest, knnbonusTestMarker, k = 1)
attach(forbonusTest)
confusionMatrix(knnbonusmodel, grouping)

#knn model with k = 3
set.seed(90)
knn3bonusmodel <- knn(knnbonusTrain, knnbonusTest, knnbonusTestMarker, k = 3)
confusionMatrix(knn3bonusmodel, grouping)

#knn model with k = 5
set.seed(90)
knn5bonusmodel <- knn(knnbonusTrain, knnbonusTest, knnbonusTestMarker, k = 5)
confusionMatrix(knn5bonusmodel, grouping)

#cross validation
set.seed(90)
knnbonusmodelcv <- train(
  grouping ~ ., 
  data = forbonus,
  method = "knn",
  trControl = trainControl(
    method = "cv", 
    number = 10
))

#Run test data thru cross-validated model
knnbonuscvprobs <- predict(knnbonusmodelcv, forbonusTest)
confusionMatrix(knnbonuscvprobs, forbonusTest$grouping)
detach(forbonusTest)

# Question 3

## How well can a decision tree predict what percentage of an athletic program's funding comes from subsidies?

### Decision Tree

#create version of dataset for use in trees
fortree <- p[, c(12:13, 18:21, 24, 26, 30, 51)]

#create initial tree
set.seed(90)
subsidyTree <- tree(inflation_adjusted_subsidyproportion ~., data = fortree)
summary(subsidyTree)
plot(subsidyTree)
text(subsidyTree, pretty = 0)
subsidyTree

#split data into training and test
intreeTrain <- createDataPartition(fortree$inflation_adjusted_subsidyproportion, p = 0.7, list = FALSE)
fortreeTrain <- fortree[intreeTrain,]
fortreeTest <- fortree[-intreeTrain,]

#train model on training data
treemodel <- tree(inflation_adjusted_subsidyproportion ~., data = fortreeTrain)

#cross validation
set.seed(90)
cvtree <- cv.tree(treemodel)
plot(cvtree$size, cvtree$dev, type = 'b')

#pruning tree
prunetree <- prune.tree(treemodel, best = 2)
plot(prunetree)
text(prunetree, pretty = 0)

#run test data thru tree model
treepred <- predict(treemodel, newdata = fortreeTest)
treetest <- fortree[-intreeTrain, "inflation_adjusted_subsidyproportion"]
treetest <- deframe(treetest)
plot(treepred, treetest)
abline(0, 1)
prunetree

#calculate MSE
mean((treepred - treetest)^2)

### Bagged Tree

set.seed(90)

#train bagging model
bagmodel <- randomForest(inflation_adjusted_subsidyproportion ~ ., data = fortreeTrain, mtry = 9, importance = TRUE)
bagmodel

#run test data thru bagging model
bagpred <- predict(bagmodel, newdata = fortreeTest)

#show results and calculate MSE
plot(bagpred, treetest)
abline(0, 1)
mean((bagpred - treetest)^2)

### Random Forest

set.seed(90)

#train random forest model
rfmodel <- randomForest(inflation_adjusted_subsidyproportion ~ ., data = fortreeTrain, mtry = 3, importance = TRUE)

#run test data thru random forest model
rfpred <- predict(rfmodel, newdata = fortreeTest)

#Calculate MSE
mean((rfpred - treetest)^2)

#Show variable importance
importance(rfmodel)
varImpPlot(rfmodel)

### Boosted Tree

set.seed(90)

#train boosted model
boostmodel <- gbm(inflation_adjusted_subsidyproportion ~ ., data = fortreeTrain, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

#plot variable importance in trained model
summary(boostmodel)

#closer look at top 2 variables in importance
par(mfrow = c(1,2))
plot(boostmodel, i="inflation_adjusted_ticket_sales")
plot(boostmodel, i="inflation_adjusted_ncaa_distributions")

#run test data thru boosted model
boostpred <- predict(boostmodel, newdata = fortreeTest, n.trees = 5000)

#calculate MSE
mean((boostpred - treetest)^2)

#retry boosted model with shrinkage paramater of 0.2 replacing default 0.001
set.seed(90)

#train tweaked boosted model
boostmodel2 <- gbm(inflation_adjusted_subsidyproportion ~ ., data = fortreeTrain, distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = FALSE)

#run test data thru tweaked boosted model
boostpred2 <- predict(boostmodel2, newdata = fortreeTest, n.trees = 5000)

#Calculate MSE
mean((boostpred2 - treetest)^2)
