##########################################################################
#PREAMBLE
##########################################################################
#house keeping
rm(list=ls())

#which packages to load
packages <- c('plyr', 'ggplot2', 'MASS')
lapply(packages, require, character.only = TRUE)

# Set a seed for repeatable plots
set.seed(01022020)

# Calculates the covariance matrix sigma using a
calcSigma <- function(X1,X2,l=1) {
  #initialize Sigma
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  #loop through rowas and columns
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      #calculate the kernel distances
      Sigma[i,j] <- exp(-0.5*(abs(X1[i]-X2[j])/l)^2)
    }
  }
  #return Sigma matrix
  return(Sigma)
}

##########################################################################
#ALL SET PARAMETERS
##########################################################################
#the observation noise is 0.1
sigma.n <- 0.1
#we do inference over 20 equally space option in 1D
x.star <- seq(-5,5,len=20)
#pre-calculate Gram matrix
sigma <- calcSigma(x.star,x.star)
#softmax tau
tau<-0.001
#function's smoothness
lambda<-1
#UCB exploration bonus
beta<-3
#how many runs for each simulations
nruns<-10000
#how many trials per run
ntrials<-10


##########################################################################
#SIMULATION FOR NOVELTY HEURISTIC
##########################################################################
#initialize data frame to collect which options are chosen during learning
dranknovel<-data.frame(run=1:nruns, ranks=rep(0, nruns))
#Let's get the party started with 1000 runs
for (nrun in 1:nruns){
  #we sample a target function from the GP with mean function 0
  target<-mvrnorm(1, rep(0, length(x.star)), sigma)
  #first observation is randomly sampled
  xnew<-sample(x.star)[1]
  #output for sampled option
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
  #initialize the data frame to track the actual observations
  datobserve<-data.frame(x=xnew, y=ynew)
  #initialize vector to track sampled confidence rank
  trackrank<-rep(0, ntrials)
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #x observations for GP
    x <- datobserve$x
    #k_xx matrix (this is also described in our GP tutorial)
    k.xx <- calcSigma(x,x)
    #k_xxs matrix
    k.xxs <- calcSigma(x,x.star)
    #KXSX matrix
    k.xsx <- calcSigma(x.star,x)
    #kxsxs matrix
    k.xsxs <- calcSigma(x.star,x.star)
    #posterior mean of the GP
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y
    #posterior covariance matrix of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs
    #get predictive standard deviations
    sig<-sqrt(diag(cov.f.star))
    #for utility, subtract max to avoid overflow
    util<-sig-max(sig)
    #softmax of probabilities
    prob<-exp(util/tau)/sum(exp(util/tau))
    #sample a new observation proportionally to the softmax probs
    xnew<-sample(x.star, 1, prob=prob)
    #get an ouptut plus observation noise
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
    #concatenate the new observations to the old ones
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew))
    #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew]
  }
  dranknovel$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}


##########################################################################
#SIMULATION FOR UPPER CONFIDENCE BOUND SAMPLER
##########################################################################
#initialize data frame to collect which options are chosen during learning
drankucb<-data.frame(run=1:nruns, ranks=rep(0, nruns))
#Let's get the party started with 1000 runs
for (nrun in 1:nruns){
  #we sample a target function from the GP with mean function 0
  target<-mvrnorm(1, rep(0, length(x.star)), sigma)
  #first observation is randomly sampled
  xnew<-sample(x.star)[1]
  #output for sampled option
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
  #initialize the data frame to track the actual observations
  datobserve<-data.frame(x=xnew, y=ynew)
  #initialize vector to track sampled confidence rank
  trackrank<-rep(0, ntrials)
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #x observations for GP
    x <- datobserve$x
    #k_xx matrix (this is also described in our GP tutorial)
    k.xx <- calcSigma(x,x)
    #k_xxs matrix
    k.xxs <- calcSigma(x,x.star)
    #KXSX matrix
    k.xsx <- calcSigma(x.star,x)
    #kxsxs matrix
    k.xsxs <- calcSigma(x.star,x.star)
    #posterior mean of the GP
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y
    #posterior covariance matrix of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs
    #get predictive standard deviations
    ucb<-f.bar.star+beta*sqrt(diag(cov.f.star))
    #for utility, subtract max to avoid overflow
    util<-ucb-max(ucb)
    #softmax of probabilities
    prob<-exp(util/tau)/sum(exp(util/tau))
    #sample a new observation proportionally to the softmax probs
    xnew<-sample(x.star, 1, prob=prob)
    #get an ouptut plus observation noise
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
    #concatenate the new observations to the old ones
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew))
    #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew]
  }
  drankucb$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}


##########################################################################
#SIMULATION FOR COMPLEXITY APPROXIMATION
##########################################################################
#initialize data frame to collect which options are chosen during learning
drankcomplex<-data.frame(run=1:nruns, ranks=rep(0, nruns))
#Let's get the party started with 1000 runs
for (nrun in 1:nruns){
  #we sample a target function from the GP with mean function 0
  target<-mvrnorm(1, rep(0, length(x.star)), sigma)
  #first observation is randomly sampled
  xnew<-sample(x.star)[1]
  #output for sampled option
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
  #initialize the data frame to track the actual observations
  datobserve<-data.frame(x=xnew, y=ynew)
  #initialize vector to track sampled confidence rank
  trackrank<-rep(0, ntrials)
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #x observations for GP
    x <- datobserve$x
    #k_xx matrix (this is also described in our GP tutorial)
    k.xx <- calcSigma(x,x)
    #k_xxs matrix
    k.xxs <- calcSigma(x,x.star)
    #KXSX matrix
    k.xsx <- calcSigma(x.star,x)
    #kxsxs matrix
    k.xsxs <- calcSigma(x.star,x.star)
    #posterior mean of the GP
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y
    #posterior covariance matrix of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs
    #get predictive standard deviations
    sig<-sqrt(diag(cov.f.star))
    #initialize utility
    util<-rep(0, length(sig))
    for (sims in seq_along(util)){
      datobservesimulated<-rbind(datobserve, data.frame(x=x.star[sims], y=f.bar.star[sims]))
      x <- datobservesimulated$x
      #k_xx matrix (this is also described in our GP tutorial)
      k.xx <- calcSigma(x,x)
      #k_xxs matrix
      k.xxs <- calcSigma(x,x.star)
      #KXSX matrix
      k.xsx <- calcSigma(x.star,x)
      #kxsxs matrix
      k.xsxs <- calcSigma(x.star,x.star)
      #posterior mean of the GP
      f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobservesimulated$y
      #posterior covariance matrix of the GP
      cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs
      #get predictive standard deviations
      signew<-sqrt(diag(cov.f.star))
      util[sims]<-sum(sig)-sum(signew)
    }
    
    #for utility, subtract max to avoid overflow
    util<-util-max(util)
    #softmax of probabilities
    prob<-exp(util/tau)/sum(exp(util/tau))
    #sample a new observation proportionally to the softmax probs
    xnew<-sample(x.star, 1, prob=prob)
    #get an ouptut plus observation noise
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n)
    #concatenate the new observations to the old ones
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew))
    #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew]
  }
  drankcomplex$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}


##########################################################################
#DENSITY HISTOGRAMS OF SAMPLED MEAN RANKS
##########################################################################
#plot for complexity heuristic
p1<-ggplot(drankcomplex, aes(x=ranks)) + 
  #histogram
  geom_histogram(fill="#7bc5e0")+xlim(c(1,5))+theme_classic()+scale_y_continuous(expand = c(0,0))+
  #style
  theme(text = element_text(size=25,  family="calibri"))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Complexity Approximation")
p1

#plot for novelty heuristic
p2<-ggplot(dranknovel, aes(x=ranks)) + 
  #histogram
  geom_histogram(fill="#0270bb")+xlim(c(1,5))+theme_classic()+scale_y_continuous(expand = c(0,0))+
  #styles
  theme(text = element_text(size=25,  family="calibri"))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Novelty Approximation")
p2


#plot for ucb
p3<-ggplot(drankucb, aes(x=ranks)) + 
  #histogram
  geom_histogram(fill="purple")+xlim(c(0,20))+theme_classic()+scale_y_continuous(expand = c(0,0))+
  #styles
  theme(text = element_text(size=25,  family="calibri"))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Upper Confidence Bounds")
p3

#save all plots:
png("sampling1.png", width=500, height=300)
p1
dev.off()

png("sampling2.png", width=500, height=300)
p2
dev.off()

png("sampling3.png", width=500, height=300)
p3
dev.off()
#END