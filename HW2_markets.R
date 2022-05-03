##################################
##Argimiro Arratia @2022 Computational Finance
##Rlab3c Task  Causality Analysis for World Markets
##################################
library(xts); library(quantmod); library(plyr)
library("MSBVAR"); library("vars")
##WARNING: MSBVAR is deprecated from CRAN. I provide old version which must be manually installed. 
## But it may not install due to its old bit configuration.
##Alternative use grangertest function in lmtest
##To only get the p-value save results in enviroment var
# GT<-grangertest(gnp~m1, order=3,data=USMoney)
# GT$`Pr(>F)`[2]
library("lmtest") ##load required library for grangertest

mktsym<- c("^BSESN", "^BVSP","^FTSE","^GDAXI","^GSPC","^HSCE", 
 "^IBEX","^JKSE","^MXX","^N225","^TWII","^VLIC","^VIX")
marktCtry<-c("India", "Brazil","UK","Germany","USA","China-Shanghai",
             "Spain","Indonesia","Mexico","Japan","Taiwan","VLIC","VIX")

dateI="1999-01-01"; dateF="2020-04-30"
#Get data from yahoo finance
data.env<-new.env()
l_ply(mktsym, function(sym) try(getSymbols(sym,src="yahoo",from=dateI,to=dateF,env=data.env),silent=T))

##Optional: save in disc historic data for this portfolio
saveRDS(data.env,file="WorldMarkts99_20.RDS")

##Start HW1 from here
### Get data from disc
data.env <- readRDS("../data/2/WorldMarkts99_20.RDS")
markets <- ls(data.env)
##to extract one market data from data.env:
## ibex <- get("IBEX",data.env)

### loop through to get Ad.Close, compute weekly return and merge all stocks and treat as xts objects
returns <- xts()
per<- "weekly" ## "monthly", "daily"##period of sampling
for(i in seq_along(markets)) {
  sym <- markets[i]
  returns <- merge(returns, 
                  periodReturn(Ad(get(sym,envir=data.env)),period=per,type = "log"))
   #periodReturn(Op(get(sym,envir=data.env)),period=per,type = "arithmetic"))
}

##Resolve NA's. Below a sloppy solution. Do something more intelligent
returns[is.na(returns)]<-0 ##imputation of 0 ##or impute previous non 0 return
colnames(returns) <- paste(markets,".ret",sep="")

##Extract the epoch to analysis
dI="2001-01-01"; dF="2003-01-01"
Retp <- returns[paste(dI,"/",dF,sep=""),]

##Granger causality test for lag 1
h=1
granger.test(Retp,h)

##run for all lags h=1,...,4 (use lapply)
caus<- lapply(1:4,function(h){granger.test(Retp,h)})
caus

##Tabulate results: row (Cause) --> column (effect), 
## each entry a 4-vector of {0,1} indicating causality (1) or not(0) for each lag
##Example
##          |"India" | "Brazil"| "UK"   | ...
##"USA"     (1,0,1,1)  (0,1,1,0)
##"Brazil"   (0,0,1,1)
## 1st entry means USA --> India at lags 1,3,4

##Do the same analysis for volatilities series of each market
## Enough to consider the variance as  EMA of square returns
## Do the analysis for week and month sampling periods
## Optional: Apply  rolling windows with first year of data and step forward one period (week or month) 


