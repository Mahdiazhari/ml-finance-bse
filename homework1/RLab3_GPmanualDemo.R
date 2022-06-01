##########################################################
##Argimiro Arratia @2021,2022 Computational Finance
# Demo of Gaussian process regression with R
# https://www.r-bloggers.com/gaussian-process-regression-with-r/
# original source by James Keirstead (2012), 
# Revised and expanded (2019) argimiro@2019 : fig2b, gg originals did not work;
## -added kernel funct; show some sample functions in different colour
##########################################################

# Chapter 2 of Rasmussen and Williams's book `Gaussian Processes
# for Machine Learning' provides a detailed explanation of the
# math for Gaussian process regression. This Gist is a brief demo
# of the basic elements of Gaussian process regression, as
# described on pages 13 to 16.


# Load in the required libraries for data manipulation
# and multivariate normal distribution
require(MASS)
require(plyr)
require(reshape2)
require(ggplot2)

## Set working directory to current script's location.
fileloc <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(fileloc)
rm(fileloc)

# Set a seed for repeatable plots
set.seed(12345)

## Kse1 sqr. exp. kernel simple (only parameter l)
## Kernel function (square exponential kernel with parameter: l)
##a y b son vectores
Kse1 <- function(a,b,l=1){
        sqdist <- sum((a-b)^2)
        return(exp(-0.5*sqdist/l^2))
}
#######################################

## Kse2 sqr. exp. kernel with param: l, sigma, p
Kse2 <- function(a,b,l=1,sigma=1,p=2){
  sqdist <- sum(abs(a-b)^p)
  return(sigma^2*exp(-0.5*sqdist/l^2))
}
#######################################

# Calculates the covariance matrix sigma using 
# kernel function K
K <- Kse1

#
# Parameters:
#	X1, X2 = vectors, each entry d-dimensional
# 	l = the scale length parameter
# Returns:
# 	a covariance matrix
calcSigma <- function(X1,X2,l=1) {
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      Sigma[i,j] <- K(X1[i],X2[j],l=l)  
    }
  }
  return(Sigma)
}

# 1. Plot some sample functions from the Gaussian process
# as shown in Figure 2.2(a)

# Define the points at which we want to define the functions
x.star <- seq(-5,5,len=50)

# Calculate the covariance matrix
sigma <- calcSigma(x.star,x.star,l=0.5)

# Generate a number of functions from the process
n.samples <- 3
values <- matrix(rep(0,length(x.star)*n.samples), ncol=n.samples)
for (i in 1:n.samples) {
  # Each column represents a sample from a multivariate normal distribution
  # with zero mean and covariance sigma
  values[,i] <- mvrnorm(1, rep(0, length(x.star)), sigma)
}
values <- cbind(x=x.star,as.data.frame(values))
values <- melt(values,id="x")

dev.off() ##to reset graphics state. Avoids ggplot getting mess-up

# Plot the result
fig2a <- ggplot(values,aes(x=x,y=value)) +
  geom_rect(xmin=-Inf, xmax=Inf, ymin=-2, ymax=2, fill="grey80") +
  #geom_line(aes(group=variable),colour="blue") +
  geom_line(aes(group=variable,colour=variable)) +
  theme_bw() +
  scale_y_continuous(lim=c(-2.5,2.5), name="output, f(x)") +
  xlab("input, x")

fig2a

# 2. Now let's assume that we have some known data points;
# this is the case of Figure 2.2(b). In the book, the notation 'f'
# is used for f$y below.  I've done this to make the ggplot code
# easier later on.
f <- data.frame(x=c(-4,-3,-1,0,2),
                y=c(-2,0,1,2,-1))

# Calculate the covariance matrices
# using the same x.star values as above
x <- f$x
k.xx <- calcSigma(x,x)
k.xxs <- calcSigma(x,x.star)
k.xsx <- calcSigma(x.star,x)
k.xsxs <- calcSigma(x.star,x.star)


# These matrix calculations correspond to equation (2.19)
# in the book.
f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs

##Generate function values f.star (corresponding to x.star inputs) by doing
##multivariate Gaussian sampling with mean and covariance from above equation (obtained from posterior distribution)
# This time we'll plot more samples.  We could of course
# simply plot a +/- 2 standard deviation confidence interval
# as in the book but I want to show the samples explicitly here.
n.samples <- 50
values <- matrix(rep(0,length(x.star)*n.samples), ncol=n.samples)
for (i in 1:n.samples) {
  values[,i] <- mvrnorm(1, f.star.bar, cov.f.star)
}
values <- cbind(x=x.star,as.data.frame(values))
values <- melt(values,id="x")

# Plot the results including the mean function (f.star.bar)
# and constraining data points
## warning: f.star.bar is a single column matrix and ggplot is expecting a vector.
## y=f.star.bar[,1] to cast the matrix column to a vector.
fig2b <- ggplot() +
  geom_line(data=values, aes(x=x,y=value, group=variable), colour="grey80") +
  geom_line(data=NULL,aes(x=x.star,y=f.star.bar[,1]),colour="red", size=1) +
  ## show 2 of the 50 sample functions (in blue & green)
  geom_line(data=NULL,aes(x=x.star,y=values$value[1:50]),colour="blue", size=1) +
  geom_line(data=NULL,aes(x=x.star,y=values$value[51:100]),colour="green", size=1) +
  ###
  geom_point(data=f,aes(x=x,y=y)) +
  theme_bw() +
  scale_y_continuous(lim=c(-3,3), name="output, f(x)") +
  xlab("input, x")

fig2b

##To view all point predictions at x.star, which correspond to values f.star.bar (the mean of distrib)
pred = cbind(x.star,f.star.bar); #View(pred)
##To view one point prediction x.s given f={(x,y)} training data:
## Apply lines 90-105, f can be more pairs (x,y),  with x.star <- x.s (a scalar). 
##The output f.star.bar, cov.f.star will be two scalars:the function value and variance at x.s
## x.s is the info we know: e.g. P/E at time t and want f.star.bar the return (Another example: Volume-->Return, Previous returns--> Next return)


# 3. Now assume that each of the observed data points have some
# normally-distributed noise.
# The standard deviation of the noise
sigma.n <- 0.1

# Recalculate the mean and covariance functions
f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%f$y
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs

# Recalculate the sample functions
values <- matrix(rep(0,length(x.star)*n.samples), ncol=n.samples)
for (i in 1:n.samples) {
  values[,i] <- mvrnorm(1, f.bar.star, cov.f.star)
}
values <- cbind(x=x.star,as.data.frame(values))
values <- melt(values,id="x")

# Plot the result, including error bars on the observed points
gg <- ggplot() +
  geom_line(data=values, aes(x=x,y=value,group=variable), colour="grey80") +
  geom_line(data=NULL,aes(x=x.star,y=f.bar.star[,1]),colour="red", size=1) +
  ## show 2 of the 50 sample functions (in blue & green)
  geom_line(data=NULL,aes(x=x.star,y=values$value[1:50]),colour="blue", size=1) +
  geom_line(data=NULL,aes(x=x.star,y=values$value[51:100]),colour="green", size=1) +
  ###
  geom_errorbar(data=f,aes(x=x,y=NULL,ymin=y-2*sigma.n, ymax=y+2*sigma.n), width=0.2) +
  geom_point(data=f,aes(x=x,y=y)) +
  theme_bw() +
  scale_y_continuous(lim=c(-3,3), name="output, f(x)") +
  xlab("input, x")

gg

####### TO DO (HW)
## 1. Modify the squared exponential covariance function to include
##   the signal and noise variance parameters, in addition to the length scale l (shown here)
## 2. Use Cholesky decomposition to speed up code (Algorithm 2.1, p 19 Rasmussen-Williams)
## Hint: check chol {base} function
