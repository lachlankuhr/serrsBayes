library(tidyverse)

full_log_posterior <- function(a_p, a_leafs, x, spcSig, nObs, priorMinAmp, priorMaxAmp, ampPriorMin, ampPriorMax, response_std, loc = 300, scale = 25) {
  # Get logLikelihood across all runs in batch
  logLikeAllData <- 0
  for (i in 1:nObs) {
    # Get the leaf prediction
    a_i <- a_leafs[i]
    # Get the spectrum from the leaf
    y_i <- spcSig[i,]
    
    # Calculate the log-likelihood
    y_pred <- a_i * exp((-1/2)*(((x-loc)/scale))^2)
    loglikes <- dnorm(y_i, y_pred, response_std, log = TRUE)
    
    # Sum over all runs in batch
    logLikeAllData <- logLikeAllData + sum(loglikes)
  }
  
  # Add on prior for proposal
  logPosterior <- logLikeAllData + log_prior(a_p, ampPriorMin, ampPriorMax)
  # Add on prior for d_i's
  for (i in 1:nObs) {
    a_i <- a_leafs[i]
    logPosterior <- logPosterior + dnorm(a_p - a_i, 0, 500, log=TRUE)
  }
  
  # Return the result
  return(logPosterior)
}

# Log likelihood of the data
log_likelihood_normal <- function(amplitude, x, y, response_std, loc = 300, scale = 25) {
  y_pred <- amplitude * exp((-1/2)*(((x-loc)/scale))^2)
  loglikes <- dnorm(y, y_pred, response_std, log = TRUE)
  return(sum(loglikes))
}

# Assuming uniform prior 
log_prior <- function(amplitude, priorMinAmp, priorMaxAmp) { 
  return(dunif(amplitude, priorMinAmp, priorMaxAmp, log=TRUE))
}

log_posterior <- function(amplitude, x_t, y_t, priorMinAmp, priorMaxAmp, response_std) {
  return(log_likelihood_normal(amplitude, x_t, y_t, response_std) + 
           log_prior(amplitude, priorMinAmp, priorMaxAmp))
}

# Calculate the ESS
# Used for selecting temperatures adaptively
calc_ESS <- function(new_gamma, old_gamma, weights, log_like) {
  # By notes
  log_weights <- log(weights) + (new_gamma - old_gamma) * 
    log_like
  
  # Numerically stabilise before exponentiating
  log_weights <- log_weights - max(log_weights)
  weights <- exp(log_weights)
  
  # Normalise
  weights <- weights / sum(weights)
  
  # Return the ESS
  return(1/sum(weights^2))
}

# Calculate MH ratio
calculate_mh_ratio <- function(amplitude_p, amplitude_c, x, y, minAmp, maxAmp, priorMinAmp, priorMaxAmp, response_std = 1) {
  num <- log_posterior(amplitude_p, x, y, priorMinAmp, priorMaxAmp, response_std) +
    dunif(amplitude_c, minAmp, maxAmp, log=TRUE)
  den <- log_posterior(amplitude_c, x, y, priorMinAmp, priorMaxAmp, response_std) +
    dunif(amplitude_p, minAmp, maxAmp, log=TRUE)
  return(min(1, exp(num - den)))
}

## Setup program configurations
N <- 4000 # number of particles
min_ESS <- N/2 # minimum effective sample size
mcmc_iterations <- 10 # number of iterations for each particle to take in the MCMC move step

nObs <- 2^3
## Set the true values of beta_0 and beta_1
#amplitudes <- c(100, 102, 104, 106, 108, 110, 112, 114)
amplitudes <- rnorm(nObs, 2500, 500)
amplitudes <- sort(amplitudes)
# Generate random x values and produce the corrosponding linear regression response values
x <- 100:500
D <- length(x)
# Assuming response is noramlly distriubted with sd = 1
response_std <- 20

spcSig <- matrix(nrow=nObs, ncol=D)

for (i in 1:length(amplitudes)) {
  spcSig[i,] <- amplitudes[i] * exp((-1/2)*(((x-300)/25))^2) + rnorm(D, 0, response_std)
}

# Plot
plot(range(x), range(spcSig), type='n', xlab="Wavenumbers", ylab="Intensity")
for (i in 1:nObs) {
  lines(x, spcSig[i,], col=i, lty=i, lwd=2)
}
title(main="Synthetic data with batch effects")
abline(h=0,lty=2)

dncParallel <- matrix(nrow=N, ncol=nObs + 2)

# Setup plots
par(mfrow=c(4,2))

for (spectrumIndex in 1:nObs) {

y <- spcSig[spectrumIndex,]

# Set prior to be standard normal and initialise weights as 1/N
priorMinAmp <- 0
priorMaxAmp <- 10000
theta_smc <- list(particles = runif(N, priorMinAmp, priorMaxAmp),
                  weights = rep(1, N)/N,
                  log_like = rep(0, N))
for (i in 1:N) {
  theta_smc$log_like[i] <- log_likelihood_normal(theta_smc$particles[i], x, y, response_std)
}

# Initialise temperature
gamma_t = 0
temp_hist = c(gamma_t)

while (gamma_t < 1) {
  # Determine gamma_{t+1}
  ess1 <- calc_ESS(1, gamma_t, theta_smc$weights, theta_smc$log_like)
  
  if (ess1 > N/2) {
    new_gamma = 1
  } else {
    bi_section = uniroot(
      function(new_gamma) calc_ESS(new_gamma, gamma_t, theta_smc$weights, theta_smc$log_like) - N/2,
      lower = gamma_t, 
      upper = 1,
      tol = .Machine$double.eps^2 # extremely low tolerance to get N/2 exactly
    )
    new_gamma <- bi_section$root
  }
  print(new_gamma)
  temp_hist = c(temp_hist, new_gamma)
  
  # Reweighting
  log_likelihoods <- rep(0, N)
  for (i in 1:N) {
    log_likelihoods[i] <- log_likelihood_normal(theta_smc$particles[i], x, y, response_std)
  }
  log_weights <- log(theta_smc$weights) + (new_gamma - gamma_t) * 
    log_likelihoods
  
  # Numerically stabilise before exponentiating
  log_weights <- log_weights - max(log_weights)
  theta_smc$weights <- exp(log_weights)
  
  # Normalise
  theta_smc$weights <- theta_smc$weights / sum(theta_smc$weights)
  
  # Return the ESS
  ESS = 1/sum(theta_smc$weights^2)
  
  print(ESS)
  
  ## Resample
  resample <- sample(1:N, N, replace = TRUE, prob = theta_smc$weights)
  theta_smc$particles <- theta_smc$particles[resample]
  theta_smc$log_like <- theta_smc$log_like[resample]
  theta_smc$weights <- rep(1,N)/N
  
  ## Move step
  minAmp <- min(theta_smc$particles)
  maxAmp <- max(theta_smc$particles)
  
  for (i in 1:N) {
    for (r in 1:mcmc_iterations) { 
      # Current 
      amplitude_c <- theta_smc$particles[i]
      
      # Proposal
      # Multivariate normal random walk with covariance given by population of particles 
      amplitude_p <- runif(1, minAmp, maxAmp)
      
      # Calculate MH ratio
      mh <- calculate_mh_ratio(amplitude_p, amplitude_c, x, y, minAmp, maxAmp, priorMinAmp, priorMaxAmp, response_std)
      
      # Move particles
      if (runif(1) < mh) {
        theta_smc$particles[i] <- amplitude_p
        theta_smc$log_like[i] <- log_likelihood_normal(amplitude_p, x, y, response_std)
      }
    }
  }
  
  # Update gamma_t
  gamma_t = new_gamma
}

## Resample
resample <- sample(1:N, N, replace = TRUE, prob = theta_smc$weights)
theta_smc$particles <- theta_smc$particles[resample]
theta_smc$log_like <- theta_smc$log_like[resample]
theta_smc$weights <- rep(1,N)/N

dncParallel[,spectrumIndex] <- theta_smc$particles
hist(theta_smc$particles, breaks = 15, main = paste("Distribution Of Amplitude - Spectrum", spectrumIndex), xlab = "Amplitude")
abline(v = amplitudes[spectrumIndex], col="red", lwd=3, lty=2)
}

for (i in 1:N) {
  a_leafs <- dncParallel[i, 1:nObs]
  min_leaf <- min(a_leafs)
  max_leaf <- max(a_leafs)
  amplitude_p <- runif(1, min_leaf, max_leaf)
  dncParallel[i, nObs + 1] = amplitude_p
  postProp <- full_log_posterior(amplitude_p, a_leafs, x, spcSig, nObs, priorMinAmp, priorMaxAmp, min_leaf, max_leaf, response_std)
  productLeafPost = 0
  for (obs in 1:nObs) {
    productLeafPost <- productLeafPost + log_posterior(dncParallel[i, obs], x, spcSig[obs,], priorMinAmp, priorMaxAmp, response_std)
  }
  dProb <- dunif(amplitude_p, min_leaf, max_leaf)
  dncParallel[i, nObs + 2] = exp(postProp - productLeafPost) * (1 / dProb)
}

dncParallel[, nObs + 2] = dncParallel[, nObs + 2] / sum(dncParallel[, nObs + 2])
#plot(dncParallel[, nObs + 2])
sum(dncParallel[, nObs + 2] * dncParallel[, nObs + 1])

dist <- sample(dncParallel[, nObs + 1], 20000, replace = TRUE, prob = dncParallel[, nObs + 2])
par(mfrow=c(1,1))
hist(dist, breaks = 15, main = "Distribution Of Amplitude - Merged", xlab = "Amplitude")
