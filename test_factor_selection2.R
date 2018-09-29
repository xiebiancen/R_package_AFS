rm(list=ls());
library(AFNSv3)
set.seed(12345)
load("combination.RData");
dat=read.csv("US_DataM.csv",header=T);
dat=dat[217:348,-1]/1200;
relevant_factor=combination[[2]]$relevant_factor
irrelevant_factor=combination[[2]]$irrelevant_factor
result=MCMC_real_data_analysis(relevant_factor = relevant_factor,
irrelevant_factor = irrelevant_factor,
dat = dat,
nu = 15,
n0 = 1000,
n1 = 7000,
J1 = 2,
J2 = 2,
tp = 0.8,
B=3)

save(result,file="result2.RData");
