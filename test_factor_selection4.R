rm(list=ls());
library(AFNSv3)
set.seed(12345)
load("result4.RData")
load("combination.RData");
dat=read.csv("US_DataM.csv",header=T);
dat=dat[349:552,-1]/1200;

relevant_factor=result$Spec$relevant_factor
irrelevant_factor=result$Spec$irrelevant_factor
result=MCMC_with_model_specific_prior(result = result,
relevant_factor = relevant_factor,
irrelevant_factor = irrelevant_factor,
dat = dat,
nu = 10,
n0 = 1000,
n1 = 7000,
J1 = 2000,
J2 = 2000,
tp = 0.8,
B=6)


save(result,file="result4.RData");
