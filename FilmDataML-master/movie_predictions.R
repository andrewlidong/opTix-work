# ===============================================================================================
# Andrew Dong, advised by Preetam Dutta
# OpTix Group
# Research into ML techniques for predictive analysis on films, week 1

# some libraries we'll be using

library(e1071)
library(glmnet)
library(MASS)
library(vcd)
library(car)
library(VIM)
library(mice)
library(PerformanceAnalytics)
library(tabplot)
library(ggplot2)
library(googleVis)
library(dplyr)
library(Rcpp)
library(neuralnet)
library(cat)
library(class)
library(rpart)
library(randomForest)
library(gbm)
library(ROCR)

# ===============================================================================================

setwd("/home/andrew/Desktop/OpTixGroup/FirstProject")
movies = read.csv("filmdat.csv")

# ===============================================================================================

# Exploratory Data Analysis

summary(movies)
head(movies)
nrow(movies)
sapply(movies, sd)

# ===============================================================================================

# Preliminary Data Preparation

# Checking for missing values
length(complete.cases(movies))
aggr(movies, numbers=T, prop=F, sortVars=T, labels=names(movies))

# Matrix plot where darkness of shade represents high values, red for missing points
matrixplot(movies, interactive=T, sortby="opening_weekend")

# Margin plot with red for missing points.  No observations of missing values 
marginplot(movies[,c("budget", "opening_weekend")])

#scattered matrix with information about imputed values
scattmatrixMiss(movies[, 1:8], interactive = F)

# complete data set.  Thanks Preetam.  

# -----------------------------------------------------------------------------------------------
# Experimenting with data sampling and generation.  Unnecessary here.  
# Sampling 100 records from movies with replacement

#index <- sample(1:nrow(movies), 100, replace=T)
#index

#moviessample <- movies[index,]
#moviessample

# Create some missing data
#moviessample[100,1] <- NA
#moviessample

#fixmovies1 <- impute(moviessample[,1:4], what='mean')
#fixmovies1

#fixmovies2 <- impute(moviessample[,1:4], what='median')
#fixmovies2

# Normalizing data by subtracting mean and dividing by SD
# x-mean(x)/standard deviation
#scalemovies <- scale(movies[,1:4])
#head(movies)
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Experimenting with principal component analysis to reduce dimensionality issues
# Not necessary with smaller data sets such as filmdat.csv, potentially useful for bigger data

#cor(movies[, 3:8])

# compute pca of those attributes showing high correlation

#pca <- prcomp(movies[,3:8], scale=T)
#summary(pca)

# PC1, PC2 and PC3 cover most of the variation
#plot(pca)
#pca$rotation

# Project first 3 records in PCA direction
#predict(pca)[1:3,]

#movies2 <- transform(movies, ratio=round(budget/opening_weekend, 2))
#head(movies2)
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Experimenting with discretizing numeric values into categories
# Equal width cuts
#segments <- 100
#maxL <- max(movies$budget)
#minL <- min(movies$budget)
#theBreaks <- seq(minL, maxL,
#                 by=(maxL-minL)/segments)
#cutbudget <- cut(movies$budget,
#                 breaks=theBreaks,
#                 include.lowest=T)
#newdata <- data.frame(orig.budget=movies$budget,
#                      cut.budget=cutbudget)
#head(newdata)
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

# Preliminary Graphical Analysis

#histogram of movies opening weekend and density, including mean value
hist(movies$opening_weekend, breaks=30)
plot(density(movies$opening_weekend))
abline(v=mean(movies$opening_weekend), lty=2)
mean(movies$opening_weekend)

#plot of budget density and normal qq
plot(density(movies$budget))
qqnorm(movies$budget)
qqline(movies$budget)

#histogram of movies by year and density
hist(movies$year)
plot(density(movies$year))

# Boxplot of Opening Weekend vs Movie Year
boxplot(opening_weekend ~ year, data=movies, col="bisque", las=2)
title("Opening Weekend vs Movie Year")
# Observe a general positive correlation (inflation).  We can correct this with inflation statistics

# Boxplot of Opening Weekend vs MPAA Ratings
reordered_mpaa_rating = with(movies, reorder(mpaa_rating, -opening_weekend, median))
boxplot(opening_weekend ~ reordered_mpaa_rating, data=movies, lwd=0.5, col="bisque", las=2)
title("Opening Weekend vs MPAA Ratings")
# Observe a high frequency of PG-13 movies having successful opening-weekends

# -----------------------------------------------------------------------------------------------
# Non-Categorical Data.  The plots were too messy.  
# opening_weekend vs runtime
#reordered_runtime = with(movies, reorder(runtime, -opening_weekend, median))
#boxplot(opening_weekend ~ reordered_runtime, data=movies, lwd=0.5, col="bisque", las=2)
#stripchart(opening_weekend ~ reordered_runtime, data=movies, vertical=T, add=T, pch=1, col='grey')
#title("Opening Weekend vs Runtime")

#boxplot(opening_weekend ~ budget, data=movies, lwd=0.5, col="bisque", las=2)
#stripchart(opening_weekend ~ budget, data=movies, vertical=T, add=T, pch=1, col='grey')
#title("Opening Weekend vs Budget")

#reordered_theaters = with(movies, reorder(theaters, -opening_weekend, median))
#boxplot(opening_weekend ~ reordered_theaters, data=movies, lwd=0.5, col="bisque", las=2)
#stripchart(opening_weekend ~ reordered_theaters, data=movies, vertical=T, add=T, pch=1, col='grey')
#title("Opening Weekend vs Theaters")
# -----------------------------------------------------------------------------------------------

# Barplot of mpaa rating distribution
categories <- table(movies$mpaa_rating)
barplot(categories, col=c('red','blue','green','yellow','brown','black','pink'))
title("MPAA Rating Distribution")

# Measuring top mpaa_ratings on opening_weekend via bubble chart
cat1 = movies %>% select(mpaa_rating, opening_weekend) %>% 
  group_by(mpaa_rating) %>% summarize(appear.count=n())
cat2 = left_join(movies, cat1, by="mpaa_rating")
cat3 = cat2 %>% select(mpaa_rating, opening_weekend, appear.count) %>%
  distinct %>% arrange(desc(appear.count))

hist(m3$appear.count, breaks=30)

Ratings <- gvisBubbleChart(cat3, idvar="mpaa_rating", 
                          xvar="appear.count", yvar="opening_weekend",
                          sizevar="appear.count",
                          #colorvar="title_mpaa_rating",
                          options=list(
                            #hAxis='{minValue:75, maxValue:125}',
                            width=1000, height=800
                          )
)
plot(Ratings)

# ===============================================================================================

# Exploratory data analysis on smaller sets

# Creating set for only numerical values
ms_all_rows = movies[, c("opening_weekend",
                         "budget", 
                         "theaters",
                         "runtime")]
ms = na.omit(ms_all_rows)

# Measuring correlation of numerical values
cor(ms)

# Plotting scatterplots of numerical values and charting correlations
plot(ms, pch='.')
chart.Correlation(ms) 

scatterplotMatrix(ms, pch=".")


# Creating set for only categorical values
msc = movies[, c("mpaa_rating",
                 "year"
)]

# A sorted image according to numerical and categorical values
tableplot(ms, sortCol="opening_weekend") # gives you a sorted image according to numerical values
tableplot(msc, sortCol="year") # gives you a sorted image according to categories

# ===============================================================================================
# Introductory Linear Regressions
# Measuring correlation via scatter plot for budget, opening_weekend, theaters.  
# Uncertain whether to use budget or logbudget.  
pairs(movies[,c(3,5,7)])
# Computing correlation matrix
cor(movies[,c(3,5,7)])

# High correlation between budget and opening_weekend, though not as high as I suspected
# Now analyzing relationship between numeric values with more specificity via regression curve

# Plotting two variables, training reg model and drawing regression line
plot(budget~opening_weekend, data=movies)
model <- lm(budget~opening_weekend, data=movies)
abline(model, col = "red")

# Training local linear model
model2 <- lowess(movies$budget~movies$opening_weekend)
lines(model2, col="blue")

# -----------------------------------------------------------------------------------------------
# Experimenting with OLAP type manipulation
# Since our data has multiple categorical attributes, we can use OLAP type manipulation
#cube <- xtabs(~mpaa_rating+year, data=movies)
#cube
#apply(cube, c("mpaa_rating"), sum)
#apply(cube, c("year"), sum)
#apply(cube, c("mpaa_rating", "year"), mean)
# -----------------------------------------------------------------------------------------------

# ===============================================================================================

# Fitting multiple linear regression model
model.saturated = lm(opening_weekend ~ budget + theaters + runtime, data = ms)
summary(model.saturated)

# Assessing Outliers
# Bonferonni p-value for extreme observations
outlierTest(model.saturated)
# Leverage Plots
leveragePlots(model.saturated)

# Influence plot showing outlyingness, leverage and influence of datapoints
influencePlot(model.saturated)

# Assessing variance inflation factors
vif(model.saturated)

# Added Variable Plots
avPlots(model.saturated)

# Penalized-likelihood criteria
AIC(model.saturated)
BIC(model.saturated)

# ===============================================================================================

# Stepwise regression used to automate the variable selection process.

# minimal model with only an intercept
model.empty = lm(opening_weekend ~ 1, data = ms)

# full model with all relevent variables
model.full = lm(opening_weekend ~ budget + theaters + runtime, data = ms)

# scope of models to search
scope = list(lower = formula(model.empty), upper = formula(model.full))

forwardAIC = step(model.empty, scope, direction = "forward", k=2)

# Summary of stepwise regression
summary(forwardAIC)
# Influence plot
influencePlot(forwardAIC)
# Variance inflation factors
vif(forwardAIC)
# Added variable Plots
avPlots(forwardAIC)
# Confidence Interval
confint(forwardAIC)

# ===============================================================================================

# Transformations on data

# boxcox transformation
bc = boxCox(forwardAIC)
# Determining best lambda value
lambda = bc$x[which(bc$y == max(bc$y))]
opening_weekend.bc = (ms$opening_weekend^lambda - 1)/lambda
model.bc = lm(opening_weekend.bc ~ theaters + budget, data=ms)

# Summary of our model with bc transformation
summary(model.bc)
# Influence Plot
influencePlot(model.bc)
# Variance Inflation
vif(model.bc)
# Added Variables
avPlots(model.bc)
# Confidence Intervals
confint(model.bc)

# ===============================================================================================

# -----------------------------------------------------------------------------------------------
# Can't really do regression with regularization due to lack of variables
# Regression with regularization

movies_relevant = movies[, c("opening_weekend", "budget", "theaters")]
mvs = na.omit(movies_relevant)
x = as.matrix(mvs[, -1])
y = mvs[, 1]

# Ridge Regression

# Fitting the ridge regression with alpha = 0 for ridge regression.
grid = 10^seq(5, -2, length = 100)
ridge.models = glmnet(x, y, alpha = 0, lambda = grid)
#2 different coefficients, estimated 100 times --once each per lambda value.
dim(coef(ridge.models))
#Inspecting the coefficient estimates.
coef(ridge.models)

# Visualizing the ridge regression shrinkage.
plot(ridge.models, xvar = "lambda", label = TRUE, main = "Ridge Regression")

# Creating training and testing sets. 70% of our data in the training, 30% in test
set.seed(0)
train = sample(1:nrow(x), 7*nrow(x)/10)
test = (-train)
y.test = y[test]

length(train)/nrow(x)
length(y.test)/nrow(x)

# Performing cross-validation in order to choose the best lambda
# 10-fold cross validation.
set.seed(0)
cv.ridge.out = cv.glmnet(x[train, ], y[train], lambda = grid, alpha = 0, nfolds = 10)
plot(cv.ridge.out, main = "Ridge Regression\n")
bestlambda.ridge = cv.ridge.out$lambda.min
bestlambda.ridge
log(bestlambda.ridge)

# MSE associated with this best value of lambda
ridge.bestlambdatrain = predict(ridge.models, s = bestlambda.ridge, newx = x[test, ])
mean((ridge.bestlambdatrain - y.test)^2)

# Refitting ridge regression on overall dataset using the best lambda value
ridge.out = glmnet(x, y, alpha = 0)
predict(ridge.out, type = "coefficients", s = bestlambda.ridge)

# Inspection of MSE for our final ridge model
ridge.bestlambda = predict(ridge.out, s = bestlambda.ridge, newx = x)
mean((ridge.bestlambda - y)^2)

# Lasso Regression

#Fitting lasso regression with alpha = 1
lasso.models = glmnet(x, y, alpha = 1, lambda = grid)

dim(coef(lasso.models))
coef(lasso.models)

# Cross-validation to choose lambda
# 10-fold cross validation.
set.seed(0)
cv.lasso.out = cv.glmnet(x[train, ], y[train], lambda = grid, alpha = 1, nfolds = 10)
plot(cv.lasso.out, main = "Lasso Regression\n")
bestlambda.lasso = cv.lasso.out$lambda.min
bestlambda.lasso
log(bestlambda.lasso)

# test MSE associated with this lambda?
lasso.bestlambdatrain = predict(lasso.models, s = bestlambda.lasso, newx = x[test, ])
mean((lasso.bestlambdatrain - y.test)^2)

#Refitting
lasso.out = glmnet(x, y, alpha = 1)
predict(lasso.out, type = "coefficients", s = bestlambda.lasso)

#Insepcting MSE
lasso.bestlambda = predict(lasso.out, s = bestlambda.lasso, newx = x)
mean((lasso.bestlambda - y)^2)
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

# Second attempt at regression

testidx <- which(1:nrow(movies)%%4==0)
moviestrain <- movies[-testidx,]
moviestest <- movies[testidx,]

model <- lm(opening_weekend~budget, data=moviestrain)
summary(model)

# Useing model to predict the output of test data
prediction <- predict(model, newdata=moviestest)
# Checking correlation with actual result
cor(prediction, moviestest$opening_weekend)

# verifying residuals are normally distributed
rs <- residuals(model)
qqnorm(rs)
qqline(rs)
shapiro.test(rs)
# they're not.  Must find a different regression or prepare our data differently.  

# Logistic Model

newcol = data.frame(isR = (moviestrain$mpaa_rating == 'R'))
traindata <- cbind(moviestrain, newcol)
head(traindata)

formula <- isR ~ budget + opening_weekend + theaters + runtime

logisticModel <- glm(formula, data=traindata, family="binomial")

# Predict the probability for test data

prob <- predict(logisticModel, newdata=moviestest, type='response')
round(prob, 3)

cv.fit <- cv.glmnet(as.matrix(moviestrain[,c(-1,-2)]),
                              as.vector(moviestrain[,3]),
                              nlambda=100, alpha=0.7,
                              family="gaussian")
plot(cv.fit)
coef(cv.fit)
prediction <- predict(cv.fit,
                      newx=as.matrix(moviestest[,c(-1,-2)]))
cor(prediction, as.vector(moviestest[,3]))

# -----------------------------------------------------------------------------------------------
# trying to fix NA values since glmnet won't accept it
#moviestrainfix <- matrix(sample(c(NA, 1:1095), 100, replace=TRUE), 10)
#d <- as.data.frame(moviestrainfix)
#d[is.na(d)] <- 0
#d

#cv.fit <- cv.glmnet(as.matrix(d[,-8]),
#                    d[,8], alpha=0.8,
#                    family="multinomial")

#try alpha = 0, Ridge regression
#fit <- glmnet(as.matrix(moviestrain[,-8]),
#              moviestrain[,8], alpha=0,
#              family="multinomial")
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

# Neural Networks

nnet_moviestrain <- moviestrain
#Binarize the categorical output
nnet_moviestrain <- cbind(nnet_moviestrain, moviestrain$mpaa_rating == 'R')
nnet_moviestrain <- cbind(nnet_moviestrain, moviestrain$mpaa_rating == 'PG-13')
nnet_moviestrain <- cbind(nnet_moviestrain, moviestrain$mpaa_rating == 'NOT RATED')
names(nnet_moviestrain)[6] <- 'R'
names(nnet_moviestrain)[7] <- 'PG_13'
names(nnet_moviestrain)[8] <- 'NOT_RATED'
nn <- neuralnet(R + PG_13 + NOT_RATED ~ budget + opening_weekend, data=nnet_moviestrain, hidden=c(3))
plot(nn)

# -----------------------------------------------------------------------------------------------
# Predictions doesn't work
#mypredict <- compute(nn, moviestest[-5])$net.result
# Put multiple binary output to categorical output
#maxidx <- function(arr) {
#  return(which(arr == max(arr)))
#}
#idx <- apply(mypredict, c(1), maxidx)
#prediction <- c('R', 'PG_13', 'NOT_RATED')[idx]
#table(prediction, moviestest$mpaa_rating)
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

# Support Vector Machine

# -----------------------------------------------------------------------------------------------
# stalls my system - unsure if more processing power is needed or there's an issue with the code
#tune <- tune.svm(mpaa_rating~ opening_weekend + budget + theaters + runtime,
#                 data=moviestrain,
#                 gamma=10^(-6:1),
#                 cost=10^(1:4))
#summary(tune)

#model <- svm(mpaa_rating~ opening_weekend + budget + theaters + runtime,
#             data = moviestrain,
#             method="C=classification",
#             kernel = "radial",
#             probability=T,
#             gamma=0.001,
#             cost=1000)
#prediction <- predict(model, moviestest)
#table(moviestest$mpaa_rating, prediction)
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

# Naive Bayes
# Handles both categorical and numeric input, but generally categorical is preferred
model <- naiveBayes(mpaa_rating ~ opening_weekend + budget + theaters + runtime, data=moviestrain)
prediction <- predict(model, moviestest[,-5])
table(prediction, moviestest[,5])

# ===============================================================================================

# K Nearest Neighbor

train_input <- as.matrix(moviestrain[,-5])
train_output <- as.vector(moviestrain[,5])
test_input <- as.matrix(moviestest[,-5])
prediction <- knn(train+input, test_input, train_output, k=5)
table(prediction, moviestest$mpaa_rating)

# ===============================================================================================

# Decision Tree

# Training the decision tree
treemodel <- rpart(mpaa_rating ~ opening_weekend + budget + theaters, data=moviestrain)
plot(treemodel)
text(treemodel, use.n=T)
# Predict using decision tree
prediction <- predict(treemodel, newdata=moviestest, type='class')
# Contingency table determines accuracy
table(prediction, moviestest$mpaa_rating)

# ===============================================================================================

# Tree Ensembles

# Random Forest

# Trains 100 trees with randomly selected attributes
model <- randomForest(mpaa_rating ~ opening_weekend + budget + theaters, data=moviestrain, nTree=500)
# Predict using the forest
prediction <- predict(model, newdata=moviestest, type='class')
table(prediction, moviestest$mpaa_rating)
importance(model)

# Gradient Boosted Trees

movies2 <- movies
newcol = data.frame(isR=(movies2$mpaa_rating=='R'))
movies2 <- cbind(movies2, newcol)
movies2[45:55,]

formula <- isR ~ opening_weekend + budget + theaters
model <- gbm(formula, data=movies2,
             n.trees=1000,
             interaction.depth=2,
             distribution="bernoulli")
prediction <- predict.gbm(model, movies2[45:55,],
                          type="response",
                          n.trees=1000)
round(prediction, 3)
summary(model)

# ===============================================================================================

# Measuring Regression Performance

# -----------------------------------------------------------------------------------------------
# debug
movies_clean <- movies[!is.na(movies$mpaa_rating),]
model <- lm(opening_weekend~ budget + theaters, data=movies_clean)
score <- predict(model, newdata=movies_clean)
actual <- movies_clean$movies
rmse <- (mean((score - actual)^2))^0.5
rmse
mu <- mean(actual)
rse <- mean((score-actual)^2 / mean(mu-actual)^2)
rse
rsquare <- 1 - rse
rsquare


# Measuring Classification Performance

nb_model <- naiveBayes(mpaa_rating ~ opening_weekend + budget + theaters, data=moviestrain)
nb_prediction <- predict(nb_model,
                         moviestrain[,-5],
                         type='raw')
score <- nb_prediction[,c("R")]
actual_class <- moviestest$mpaa_rating == 'R'
# Adding some noise to the score
# Making the score less precise to demonstrate
score <- (score + runif(length(score))) / 2
#pred <- prediction(score, actual_class)
#perf <- performance(pred, "prec", "rec")
#plot(perf)

nb_model <- naiveBayes(mpaa_rating ~ opening_weekend + budget + theaters, data=moviestrain)
nb_prediction <- predict(nb_model, moviestest[,-5], type='raw')
score <- nb_prediction[,c("R")]
actual_class <- moviestest$mpaa_rating == 'R'
# mixing noise
# reducing precision
score <- (score + runif(length(score))) / 2
#pred <- prediction(score, actual_class)
#perf <- performance(pred, "tpr", "fpr")
#auc <- performance(pred, "auc")
#auc <- unlist(slot(auc, "y.values"))
#plot(perf)
#legend(0.6,0.3,c(c(paste('AUC is', auc)),"\n"),
#       border="white", cex=1.0,
#       box.col = "white")

# Plotting cost curve to find the best cutoff
# Assigning cost for false +ve and false -ve
# perf<- performance(pred, "cost", cost.fp=4, cost.fn=1)
# plot(perf)
# -----------------------------------------------------------------------------------------------
# ===============================================================================================

