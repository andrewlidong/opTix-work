# ===============================================================================================
# Andrew Dong, advised by Preetam Dutta
# OpTix Group
# Using Neural Networks for point-estimation

# Some libraries we'll be using

library(mlbench)
library(nnet)
library(caret)
library(neuralnet)
library(boot)
library(plyr) 
library(clusterGeneration)


# ===============================================================================================

setwd("/home/andrew/Desktop/OpTixGroup/ThirdProjectNN")
movies = read.csv("filmdat.csv")

# ===============================================================================================

# inspecting the range, which is from 162 - 208806300

summary(movies$opening_weekend)

# modeling linear regression

lm.fit <- lm(opening_weekend ~ budget + log_budget + runtime + mpaa_rating + theaters, data=movies)

lm.predict <- predict(lm.fit)

#MSE: 219260555627404
mean((lm.predict - movies$opening_weekend)^2)

plot(movies$opening_weekend, lm.predict,
     main="Linear regression predictions vs actual",
     xlab="Actual")

# modeling neural network
nnet.fit <- nnet(opening_weekend/(208806300-162) ~ budget + log_budget + runtime + mpaa_rating + theaters, data=movies, size=2)

# multiply by 208806138 to restore original scale
nnet.predict <- predict(nnet.fit) * 208806138

# MSE: 904336090477518
mean((nnet.predict - movies$opening_weekend)^2)

plot(movies$opening_weekend, nnet.predict,
     main = "Neural network predictions vs actual",
     xlab = "Actual")

mygrid <- expand.grid(.decay=c(0.5, 0.1), .size=c(4,5,6))
nnetfit <- train(opening_weekend/208806138 ~ budget + log_budget + runtime + mpaa_rating + theaters, data=movies, method="nnet", maxit=1000, tuneGrid=mygrid, trace=F) 
print(nnetfit)

lmfit <- train(opening_weekend ~ budget + log_budget + runtime + mpaa_rating + theaters, data=movies, method="lm")
print(lmfit)

# ===============================================================================================

# checking for missing data
apply(movies, 2, function(x) sum(is.na(x)))

# randomly splitting data into train and test sets, and fitting linear regression model and testing it on test set
index <- sample(1:nrow(movies), round(0.75*nrow(movies)))
trainmovies <- movies[index,]
testmovies <- movies[-index,]
lm.fit <- glm(opening_weekend ~ budget + runtime + theaters, data=trainmovies)
summary(lm.fit)
pr.lm <- predict(lm.fit, testmovies)
MSE.lm <- sum((pr.lm - testmovies$opening_weekend)^2)/nrow(testmovies)


# ===============================================================================================

# We should normalize data before training our neural network.  
# We use min-max scaling method in the interval [0,1]

keeps <- c("opening_weekend", "budget", "runtime","theaters")
movies[keeps]

normalize<-function(x){
  return((x-min(x)/max(x)-min(x)))
}

moviesNorm <- as.data.frame(lapply(movies[keeps],normalize))
trainmovies <- moviesNorm[index,]
testmovies <- moviesNorm[-index,]

n <- names(trainmovies[keeps])
f <- as.formula(paste("opening_weekend ~", paste(n[!n %in% "opening_weekend"], collapse = " + ")))
# errormessage: algorithm did not converge within stepmax
# Neural net with 2 hidden layers, with 5 and then 3 neurons
nn <- neuralnet(f,data=trainmovies,hidden=c(5,3),linear.output=T)
# algorithm did not converge
nn2 <- neuralnet(opening_weekend ~ budget + theaters + runtime, data=moviestrain, hidden=c(3))
# asking preetam for help on convergence

# the simplest model I could come up with.  

nn <- neuralnet(f, data=trainmovies[keeps], hidden = 1, threshold = 1,        
          stepmax = 1e+05, rep = 1, startweights = NULL, 
          learningrate.limit = NULL, 
          learningrate.factor = list(minus = 0.5, plus = 1.2), 
          learningrate=NULL, lifesign = "none", 
          lifesign.step = 1000, algorithm = "rprop-", 
          err.fct = "sse", act.fct = "logistic", 
          linear.output = TRUE, exclude = NULL, 
          constant.weights = NULL, likelihood = FALSE)



plot(nn)

# black lines represent connections, blue lines represent bias terms


# ===============================================================================================

# Predicting opening_weekend using neural net
# net outputs a normalized prediction so it needs to be scaled back for prediction

pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(movies$opening_weekend)-min(movies$opening_weekend))+min(movies$opening_weekend)
test.r <- (testmovies$opening_weekend)*(max(movies$opening_weekend)-min(movies$opening_weekend))+min(movies$opening_weekend)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(testmovies)

# compare MSE of linear model and neural net
print(paste(MSE.lm, MSE.nn))

# If the NNs MSE is lower, then great!  


# ===============================================================================================

# Visualization of network and lm performance

par(mfrow=c(1,2))

plot(testmovies$opening_weekend,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(testmovies$opening_weekend,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

plot(testmovies$opening_weekend,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(testmovies$opening_weekend,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))


# ===============================================================================================

# 10-fold cross-validation

set.seed(200)
lm.fit <- glm(opening_weekend ~., data=movies[keeps])
cv.glm(movies, lm.fit, K=10)$delta[1]

set.seed(450)
cv.error <- NULL
k <- 10

pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(movies[keeps]),round(0.9*nrow(movies[keeps])))
  train.cv <- moviesNorm[index,]
  test.cv <- moviesNorm[-index,]
  
  nn <- neuralnet(f,movies[keeps],hidden=c(5,2),linear.output=T)
  
  pr.nn <- compute(nn,testmovies.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(movies[keeps]$opening_weekend)-min(movies[keeps]$opening_weekend))+min(movies[keeps]$opening_weekend)
  
  test.cv.r <- (test.cv$opening_weekend)*(max(movies[keeps]$opening_weekend)-min(movies[keeps]$opening_weekend))+min(movies[keeps]$opening_weekend)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}


# ===============================================================================================

# calculate average MSE
mean(cv.error)

# boxplot
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

# ===============================================================================================

