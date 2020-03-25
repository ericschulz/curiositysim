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

# Calculates the covariance matrix sigma for a radial basis function (RBF) kernel used in Gaussian process regression
calcSigma <- function(X1,X2,l=1) {
  #initialize Sigma
  Sigma <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  #loop through rowas and columns
  for (i in 1:nrow(Sigma)) {
    for (j in 1:ncol(Sigma)) {
      #calculate the kernel distances, where l is the length-scale parameter
      Sigma[i,j] <- exp(-0.5*(abs(X1[i]-X2[j])/l)^2)
    }
  }
  #return Sigma matrix
  return(Sigma)
}

##########################################################################
#PARAMETERS
##########################################################################
#Task parameters
x.star <- seq(-5,5,len=20) #we do inference over 20 equally space option in 1D
nruns<-10000 #how many runs for each simulations
ntrials<-10 #how many trials per run

#Gaussian process paramerters
sigma.n <- 0.1 #the observation noise is 0.1
lambda<-1 #length scale
sigma <- calcSigma(x.star,x.star, l = lambda) #pre-calculate Gram matrix of the rbf kernel

#Sampling strategies
beta<-3 #UCB exploration bonus
tau<-0.001 #softmax tau



##########################################################################
#SIMULATION FOR NOVELTY HEURISTIC
##########################################################################
dranknovel<-data.frame(run=1:nruns, ranks=rep(0, nruns)) #initialize data frame to collect which options are chosen during learning

#Let's get the party started
for (nrun in 1:nruns){ #loop through runs (i.e., replications)
  #Generate the function we are trying to optimize
  target<-mvrnorm(1, rep(0, length(x.star)), sigma) #we sample a target function from the GP with mean function 0. 
  xnew<-sample(x.star)[1] #first observation is randomly sampled
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #output for sampled option, with added gaussian noise
  datobserve<-data.frame(x=xnew, y=ynew) #initialize the data frame to track the actual observations
  trackrank<-rep(0, ntrials) #initialize vector to track sampled confidence rank
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #GP inference
    x <- datobserve$x #x observations for GP
    k.xx <- calcSigma(x,x) #calculate the kernel matrix for all pairwise combinations of observed data points (x)
    k.xxs <- calcSigma(x,x.star) #calculate the kernel matrix for all pairwise combinations of observed data points (x) and the points we want to make predictions about (x.star)
    k.xsx <- calcSigma(x.star,x) #Same as above, but the other way around
    k.xsxs <- calcSigma(x.star,x.star) #calculate the kernel matrix for all pairwise combinations of points we want to make predictions about (x.star)
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y #posterior mean of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs #posterior covariance matrix of the GP
    sig<-sqrt(diag(cov.f.star)) #get predictive standard deviations
    #Figure out which stimuli sample
    util<-sig-max(sig) #compute utilities based on the predictive standard deviations. Here, we subtract max to avoid overflow in the softmax function, without affecting the final choice probabilitie
    prob<-exp(util/tau)/sum(exp(util/tau)) #softmax of probabilities
    xnew<-sample(x.star, 1, prob=prob) #sample a new observation proportionally to the softmax probs
    #Acquire new observations
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #get an ouptut plus observation noise
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew)) #concatenate the new observations to the old ones
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew] #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
  }
  #Update
  dranknovel$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}

saveRDS(dranknovel, 'data/noveltyHeuristic.Rds') #Save data

##########################################################################
#SIMULATION FOR COMPLEXITY APPROXIMATION 
#Warning: takes about 15 - 30 minutes to run
##########################################################################
drankcomplex<-data.frame(run=1:nruns, ranks=rep(0, nruns)) #initialize data frame to collect which options are chosen during learning
#Let's get the party started 
for (nrun in 1:nruns){
  #Generate the function we are trying to optimize
  target<-mvrnorm(1, rep(0, length(x.star)), sigma) #we sample a target function from the GP with mean function 0. 
  xnew<-sample(x.star)[1] #first observation is randomly sampled
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #output for sampled option, with added gaussian noise
  datobserve<-data.frame(x=xnew, y=ynew) #initialize the data frame to track the actual observations
  trackrank<-rep(0, ntrials) #initialize vector to track sampled confidence rank
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #GP inference
    x <- datobserve$x #x observations for GP
    k.xx <- calcSigma(x,x) #calculate the kernel matrix for all pairwise combinations of observed data points (x)
    k.xxs <- calcSigma(x,x.star) #calculate the kernel matrix for all pairwise combinations of observed data points (x) and the points we want to make predictions about (x.star)
    k.xsx <- calcSigma(x.star,x) #Same as above, but the other way around
    k.xsxs <- calcSigma(x.star,x.star) #calculate the kernel matrix for all pairwise combinations of points we want to make predictions about (x.star)
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y #posterior mean of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs #posterior covariance matrix of the GP
    sig<-sqrt(diag(cov.f.star)) #predictive standard deviations
    util<-rep(0, length(sig)) #initialize utility
    for (sims in seq_along(util)){ #perform simulations based on the future implications of sampling each of the potential stimuli
      datobservesimulated<-rbind(datobserve, data.frame(x=x.star[sims], y=f.bar.star[sims])) #rbind data grame
      x <- datobservesimulated$x #New simulated observation
      k.xx <- calcSigma(x,x) #calculate the kernel matrix for all pairwise combinations of observed data points (x)
      k.xxs <- calcSigma(x,x.star) #calculate the kernel matrix for all pairwise combinations of observed data points (x) and the points we want to make predictions about (x.star)
      k.xsx <- calcSigma(x.star,x)  #Same as above, but the other way around
      k.xsxs <- calcSigma(x.star,x.star) #calculate the kernel matrix for all pairwise combinations of points we want to make predictions about (x.star)
      f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobservesimulated$y #posterior mean of the GP
      cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs #posterior covariance matrix of the GP
      signew<-sqrt(diag(cov.f.star)) # predictive standard deviations
      util[sims]<-sum(sig)-sum(signew) #Define utility as  the difference (i.e., change) in uncertainty
    }
    #Figure out which stimuli sample
    util<-util-max(util) #fsubtract max to avoid overflow
    prob<-exp(util/tau)/sum(exp(util/tau)) #softmax of probabilities
    xnew<-sample(x.star, 1, prob=prob) #sample a new observation proportionally to the softmax probs
    #acquire new observation
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #get an ouptut plus observation noise
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew)) #concatenate the new observations to the old ones
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew] #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
  }
  #update
  drankcomplex$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}

saveRDS(drankcomplex, 'data/complexityHeuristic.Rds') #Save data


##########################################################################
#SIMULATION FOR UPPER CONFIDENCE BOUND SAMPLER
##########################################################################
drankucb<-data.frame(run=1:nruns, ranks=rep(0, nruns)) #initialize data frame to collect which options are chosen during learning

#Let's get the party started 
for (nrun in 1:nruns){ #loop through runs (i.e., replications)
  #Generate the function we are trying to optimize
  target<-mvrnorm(1, rep(0, length(x.star)), sigma) #we sample a target function from the GP with mean function 0
  xnew<-sample(x.star)[1] #first observation is randomly sampled
  ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #output for sampled option
  datobserve<-data.frame(x=xnew, y=ynew) #initialize the data frame to track the actual observations
  trackrank<-rep(0, ntrials) #initialize vector to track sampled confidence rank
  #for 10 trials in total, we'll be sampling!
  for (trials in 1:ntrials){
    #GP inference
    x <- datobserve$x #x observations for GP
    k.xx <- calcSigma(x,x) #calculate the kernel matrix for all pairwise combinations of observed data points (x)
    k.xxs <- calcSigma(x,x.star) #calculate the kernel matrix for all pairwise combinations of observed data points (x) and the points we want to make predictions about (x.star)
    k.xsx <- calcSigma(x.star,x) #Same as above, but the other way around
    k.xsxs <- calcSigma(x.star,x.star) #calculate the kernel matrix for all pairwise combinations of points we want to make predictions about (x.star)
    f.bar.star <- k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%datobserve$y #posterior mean of the GP
    cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx + sigma.n^2*diag(1, ncol(k.xx)))%*%k.xxs #posterior covariance matrix of the GP
    #Figure out which stimuli sample
    ucb<-f.bar.star+beta*sqrt(diag(cov.f.star)) #compute upper confidence bound based on combining expected outcome f.bar.star with the predictive standard deviation sqrt(diag(cov.f.star)), where beta is the exploration bonus
    util<-ucb-max(ucb) #subtract max to avoid overflow in softmax calculation
    prob<-exp(util/tau)/sum(exp(util/tau)) #softmax of probabilities
    xnew<-sample(x.star, 1, prob=prob) #sample a new observation proportionally to the softmax probs
    #Acquire new observation
    ynew<-target[x.star==xnew]+rnorm(1, 0, sigma.n) #get an ouptut plus observation noise
    datobserve<-rbind(datobserve, data.frame(x=xnew, y=ynew)) #concatenate the new observations to the old ones
    trackrank[trials]<-rank(-diag(cov.f.star), ties.method = "min")[x.star==xnew] #collect the tracked confidence ranks, we want less confidence to have lower ranks and we also break ties randomly
  }
  #Update
  drankucb$ranks[nrun]<-mean(trackrank)
  cat(paste("Run", nrun, "is done.\n"))
}

saveRDS(drankucb, 'data/ucbHeuristic.Rds') #Save data
##########################################################################
#DENSITY HISTOGRAMS OF SAMPLED MEAN RANKS
##########################################################################

#plot for novelty heuristic
p1<-ggplot(dranknovel, aes(x=ranks)) + 
  geom_histogram(fill="#0270bb")+xlim(c(1,5))+theme_classic()+scale_y_continuous(expand = c(0,0))+ #histogram
  theme(text = element_text(size=25))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Novelty Approximation") #styles
p1


#plot for complexity heuristic
p2<-ggplot(drankcomplex, aes(x=ranks)) + 
  geom_histogram(fill="#7bc5e0")+xlim(c(1,5))+theme_classic()+scale_y_continuous(expand = c(0,0))+ #histogram
  theme(text = element_text(size=25))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Complexity Approximation") #style
p2



#plot for ucb
p3<-ggplot(drankucb, aes(x=ranks)) + 
  geom_histogram(fill="purple")+xlim(c(0,20))+theme_classic()+scale_y_continuous(expand = c(0,0))+ #histogram
  theme(text = element_text(size=25))+xlab("Confidence Rank")+ylab("Counts")+ggtitle("Upper Confidence Bounds") #histogram
p3

#save all plots:
png("plots/Novelty.png", width=500, height=300)
p1
dev.off()

png("plots/Complexity.png", width=500, height=300)
p2
dev.off()

png("plots/UCB.png", width=500, height=300)
p3
dev.off()
#END