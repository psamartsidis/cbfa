rm(list=ls())
Sys.setenv("PKG_CXXFLAGS"="-fopenmp -std=c++11 -Wall -Wno-sign-compare -O2")
library(Rcpp)
sourceCpp('./src/cbfa.cpp')
load('./R/dataset_vignette.RData')



# Dataset 
nTimes = dim(y)[1]
nUnits = dim(y)[2]
y = array( y, c(nTimes,nUnits,1) )
z = array( z, c(nTimes,nUnits,1) )
w = array( w, c(nTimes,nUnits,1) )
k = array( k, c(nTimes,nUnits,1) )
n = array( n, c(nTimes,nUnits,1) )



# MCMC parameters
nThin  = 10#25
nBurn  = 500
nIter  = 500
nMCMC  = nThin*(nBurn+nIter)
tpb    = rep(0.5,7)
tpb[7] = 0.1


# Fun the algorithm
set.seed(123)
St = Sys.time()
fit = cbfa( nMCMC=nMCMC, nBurn=nBurn*nThin, nThin=nThin, window=100, P=20, y=y, n=n, k=k, w=w, z=z, Timings=Timings,
            SMAX=10, tpb_prior=tpb, tpb_nBurn=1000, nCores0=6, seed=129 )
Fn = Sys.time()
Fn-St



# Diagnostics 
save(fit,file='./R/fit_vignette.RData')
