##################################
##Argimiro Arratia @2020,2021 Computational Finance
##Rlab: GaussProc exploration: 
##  forecast of PRICE or RETURN
###### Testing Shiller&Campbell hyp: PE10 --> S&P500
###### http://computationalfinance.lsi.upc.edu
##################################
## Set working directory to current script's location.
fileloc <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(fileloc)
rm(fileloc)
library(kernlab) ##for GP
library(MASS)
#library(Metrics)##Measures of prediction error:mse, mae
library(xts) ## handling time series
library(plyr)
library(reshape2)
library(ggplot2)

set.seed(1996)

### Forecasting with GP
###EVALUATION Functs
##1. Evaluation for TARGET prediction.  
##sum of squares errors function
ssr <-function(actual,pred){
  sum((actual - pred)^2)
}
##Normalize Residual Mean Square Error (NRMSE) funct
nrmse <- function(actual,pred){
  sqrt(ssr(actual,pred)/((length(actual)-1)*var(actual)))
} 
##percentage of outperforming direct sample mean (sample expected value)
pcorrect<- function(actual,pred){
  (1-nrmse(actual,pred))*100
}

##Data
sp500 = as.xts(read.zoo('../data/SP500_shiller.csv',sep=',',header=T, format='%Y-%m-%d'))
data=sp500['1900/2012']
plot(data$SP500)

##frequency of sampling
tau=1 #data is monthly. Try tau=12 (year), tau=3 (quarterly)

##1. Target and Feature as plain Price
target <- data$SP500
sp500PE = na.omit(data$P.E10) ##feature:P/E MA(10)

##2. Target and Feature as Returns
sp500PE = diff(log(sp500PE),diff=tau)
target=diff(log(data$SP500),diff=tau)  ##compute tau-period returns
##try non-center (above) /center target (below)
target=na.trim(target-mean(na.omit(target))) ##Center the target 

##Model Inputs:
##Define matrix of features (each column is a feature)
#Features: lags 1,2,3 with (or w/o) PE10 and lags 1,2
feat = merge(na.trim(lag(target,1)),na.trim(lag(target,2)),na.trim(lag(target,3)),
             sp500PE,na.trim(lag(sp500PE,1)),na.trim(lag(sp500PE,2)),
             #add other features here,
             all=FALSE)

##add TARGET. We want to predict PRICE or RETURN
dataset = merge(feat,target,all=FALSE)
colnames(dataset) = c("lag.1", "lag.2", "lag.3",
                      "PE10","PE10.1","PE10.2",
                      #names of other features,
                      "TARGET")

##Divide data into training (75%) and testing (25%). 
T<-nrow(dataset)
p=0.75
T_trn <- round(p*T)
trainindex <- 1:T_trn
##process class sets as data frames
training = as.data.frame(dataset[trainindex,])
rownames(training) = NULL
testing = as.data.frame(dataset[-trainindex,])
rownames(testing) = NULL


## GP Kernel: vanilla is non-stationary; tanh, rbf are stationary
## Define my own Kernel (for vectors). You have 3 choices below:
##SE kernel with length scale l
l=0.003 ##0.15, 0.3
 MyKer <- function(x,y) {
  1.5*exp((sum((x-y)^2))/(-2*l^2))
 }

 MyKer <- function(x,y) {
  2*exp(sum(abs(x-y))/(-2*1.5^2)) + 1.5*sum(x*y)
}

class(MyKer) <- 'kernel'

gpfit = gausspr(TARGET~., data=training,
                type="regression",
                #kernel="tanhdot", 
                #kernel= "rbfdot", #"vanilladot",
                kernel= MyKer,  
                #kpar = list(sigma = 0.4), #list of kernel hyper-parameters for rbf
                ## if you make it constant value then does not make mle estimation of sigma
                #kpar=list(scale=2,offset=2), ##for tanh
                var = 0.003 # the initial noise variance: 0.001 default min value
)
gpfit

##build predictor (predict on test data)
GPpredict <- predict(gpfit,testing)

############Some useful methods of gausspr object gpfit 
# fitted(gpfit) #returns the fitted values
# type(gpfit) #returns type of learning: regression, classif
# error(gpfit) #returns training.error
### to see available kernels
#?rbfdot
################################################

### Evaluation of Results
actualTS=testing[,ncol(testing)] ##the true series to predict
predicTS = GPpredict

res <- list("GP"=pcorrect(actualTS,predicTS))
unlist(res)

##Check the gap between training error (mse) and testing error
train.error <- error(gpfit)  
testing.error <- mean((actualTS - predicTS)^2)
gap <- testing.error - train.error ; gap

##For visual comparison

yl=c(min(actualTS,predicTS),max(actualTS,predicTS)) #set y limits
plot(actualTS,predicTS,ylim=yl)

##Forecasting: PRICE, draw the simulations of price
#par( mfrow = c( 1, 2 ) )
#par(mar=c(2.5,2.5,2.5,2.5))
plot(actualTS,t='l',col='gray20', ylab='', xlab ='',lty=3, main='GP predictions', cex.main=0.75)
lines(GPpredict,col='green',lwd=2)
legend('bottomright',legend = c('target','GP'),col=c('gray20','green'),lty=c(3,1),cex=.7)

################ END ##############################
