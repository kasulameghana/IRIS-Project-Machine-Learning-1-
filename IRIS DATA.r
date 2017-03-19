# IRIS Data Set Machine Learning (Hello world)
install.packages("caret")
library(caret)

#Machine learning steps 

#1. Define the problem


#2. Prepare the Data

# The dataset must be loded, it is already present in the packages

library(datasets)
data("iris")
summary(iris)

# Creating a validation Dataset to test our model
# I am going to split the data in 2 parts, 80% and 20% , so that I can test my algorithms on the unused part of data

#Validation Data - used for modelings
validation_index = createDataPartition(iris$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
irisval <- iris[-validation_index,]
# use the remaining 80% of data to training and testing the models
iristrain <- iris[validation_index,]



#3. Summarizing the dataset

# I will summarize the dataset in different ways to get comfortable with it

#dimentions of dataset
dim(iris)

#dimentions of training dataset
dim(iristrain)

#dimentions of validation dataset
dim(irisval)

#Types of attributes

sapply(iristrain,class)
#All are numeric and just one is factor, which is the species names.


#different factors method 1
levels(iristrain$Species)


#different factors method 2
install.packages("sqldf")
library(sqldf)
df1 = iris
factor1 <- sqldf("select distinct Species as 'flower_type' from iristrain")
# to print the names of factors
factor1    


#First 5 rows of the data
head(iristrain)

# summarize the class distribution
percentage = prop.table(table(iristrain$Species)) * 100
cbind(freq = table(iristrain$Species), percentage=percentage)
# We can see that each class is equally distribustion. It is a normal distribustion


# statistical summary
summary(iristrain)
#All the measurements are in centimeters and the min is 0.1 and the maximum is 7.9


#4. Data exploration - graphical 
#  Exploring data to understand each attribute and the relationship between each

#split the inputs and the outputs

input <- iristrain[,1:4]
output <- iristrain[,5]

#Blox plot or whisker plot for each
# boxplot for each attribute on one image

par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(input[,i], main=names(iristrain)[i])
}

#As we can see, there are equal number of each species of flowers

table(iristrain$Species)
pie(table(iristrain$Species))


#scatter plot matrix
install.packages("ellipse")
library(ellipse)
featurePlot(x=input,y=output,plot = "ellipse")

featurePlot(x=input,y=output,plot = "box")
#This is useful to see that there are clearly different distributions of the attributes for each class value.

#Density plot of each value by the features
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=input, y=output, plot="density", scales=scales)


#5. Now it is time to create some models of the data and estimate their accuracy on unseen data.

# 3 steps:
  
#1. Set-up the test harness to use 10-fold cross validation.
#2. Build 5 different models to predict species from flower measurements
#3. Select the best model.

# 10-fold crossvalidation estimate accuracy
# This will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations of train-test splits. 
# We will also repeat the process 3 times for each algorithm with different splits of the data into 10 groups, 
# in an effort to get a more accurate estimate.

control <- trainControl(method = "cv" , number = 10)
metric <- "Accuracy"

# We are using the metric of "Accuracy" to evaluate models. This is a ratio of the number of correctly predicted instances in
# divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).
# We will be using the metric variable when we run build and evaluate each model next.

# BUILD MODELS
# This is a good mixture of simple linear (LDA), nonlinear (CART, kNN) and complex nonlinear methods (SVM, RF)
# Let's evaluate 5 different algorithms:
#1. Linear discriminant Analysis (LDA)
#2. Classification regression trees (CART)
#3. K-nearest Neighbours (kNN)
#4. SUpport Vector Machine with Linear kernel (SVM)
#5. Random Forest (RF)

#We reset the random number seed before reach run to ensure that the evaluation of each algorithm is performed using exactly the
#same data splits. It ensures the results are directly comparable.

#Let's build our five models:

install.packages("e1071")
library(e1071)
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=iristrain, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=iristrain, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=iristrain, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=iristrain, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=iristrain, method="rf", metric=metric, trControl=control)

# summarize accuracy of models

results <- resamples(list(lda = fit.lda, cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)
# We can see that the most accurate model in this case was LDA

# summarize Best MOdel
print(fit.lda)


#6. Pridiction
# LDA is the most accurate model. Now we will use that model on our validation 
# dataset to do a final check. 
# Validation set is important incase overfitting or data leak happens. BOth can result in over optimistic results.

# Running LDA directly on our validation dataset
prediction <- predict(fit.lda, validation)
confusionMatrix(prediction,validation$Species)









