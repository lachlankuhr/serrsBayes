#' Fit the model with Voigt peaks using Sequential Monte Carlo (SMC).
#'
#' @inheritParams fitSpectraSMC
#' @param mcAR target acceptance rate for the MCMC kernel
#' @param mcSteps number of iterations of the MCMC kernel
#' @importFrom methods as
#' @importFrom stats rlnorm rnorm rgamma runif cov.wt cov2cor median
#' @importFrom truncnorm rtruncnorm
#' @importFrom splines bs
#' @importFrom Matrix Matrix crossprod determinant
#' @examples 
#' wavenumbers <- seq(200,600,by=10)
#' spectra <- matrix(nrow=1, ncol=length(wavenumbers))
#' peakLocations <- c(300,500)
#' peakAmplitude <- c(10000,4000)
#' peakScale <- c(10, 15)
#' signature <- weightedLorentzian(peakLocations, peakScale, peakAmplitude, wavenumbers)
#' baseline <- 1000*cos(wavenumbers/200) + 2*wavenumbers
#' spectra[1,] <- signature + baseline + rnorm(length(wavenumbers),0,200)
#' lPriors <- list(scaG.mu=log(11.6) - (0.4^2)/2, scaG.sd=0.4, scaL.mu=log(11.6) - (0.4^2)/2,
#'    scaL.sd=0.4, bl.smooth=5, bl.knots=20, loc.mu=peakLocations, loc.sd=c(5,5),
#'    beta.mu=c(5000,5000), beta.sd=c(5000,5000), noise.sd=200, noise.nu=4)
#' result <- fitVoigtPeaksSMC(wavenumbers, spectra, lPriors, npart=50, mcSteps=1)
fitVoigtPeaksSMC_update <- function(wl, spc, lPriors, conc=rep(1.0,nrow(spc)), npart=10000, rate=0.9, mcAR=0.23, mcSteps=10, minESS=npart/2, destDir=NA, number_of_threads = 4) {
  #sourceCpp("/home/lachlan/Honours/serrsBayes/src/mixVoigt.cpp", verbose = FALSE, showOutput = FALSE)
  
  # Begin timing init
  init_start_time <- Sys.time()
  
  N_Peaks <- length(lPriors$loc.mu)
  N_WN_Cal <- length(wl)
  N_Obs_Cal <- nrow(spc)
  lPriors$noise.SS <- lPriors$noise.nu * lPriors$noise.sd^2
  print(paste("SMC with",N_Obs_Cal,"observations at",length(unique(conc)),"unique concentrations,",npart,"particles, and",N_WN_Cal,"wavenumbers."))
  
  c_c <- 0.05 # 1-c is probability at least one move
  prev_accept_rate <- 0.5 # Estimated for the first iteration. Will updated as iterations pass
  
  scale_G_mask <- 1:N_Peaks
  scale_L_mask <- (N_Peaks+1):(2*N_Peaks)
  location_mask <- (2*N_Peaks+1):(3*N_Peaks)
  beta_mask <- (3*N_Peaks+1):(4*N_Peaks)
  
  # Step 0: cubic B-spline basis (Eilers & Marx, 1996)
  ptm <- proc.time()
  Knots<-seq(min(wl),max(wl), length.out=lPriors$bl.knots)
  r <- max(diff(Knots))
  NK<-lPriors$bl.knots
  X_Cal <- bs(wl, knots=Knots, Boundary.knots = range(Knots) + c(-r,+r),
              intercept = TRUE)
  class(X_Cal) <- "matrix"
  XtX <- Matrix(crossprod(X_Cal), sparse=TRUE)
  NB_Cal<-ncol(X_Cal)
  FD2_Cal <- diff(diff(diag(NB_Cal))) # second derivatives of the spline coefficients
  Pre_Cal <- Matrix(crossprod(FD2_Cal), sparse=TRUE)
  
  # Demmler-Reinsch orthogonalisation
  R = chol(XtX + Pre_Cal*1e-9) # just in case XtX is ill-conditioned
  Rinv <- solve(R)
  Rsvd <- svd(crossprod(Rinv, Pre_Cal %*% Rinv))
  Ru <- Rinv %*% Rsvd$u
  A <- X_Cal %*% Rinv %*% Rsvd$u
  lPriors$bl.basis <- X_Cal
  lPriors$bl.precision <- as(Pre_Cal, "dgCMatrix")
  lPriors$bl.XtX <- as(XtX, "dgCMatrix")
  lPriors$bl.orthog <- as.matrix(A)
  lPriors$bl.Ru <- as.matrix(Ru)
  lPriors$bl.eigen <- Rsvd$d
  print(paste0("Step 0: computing ",NB_Cal," B-spline basis functions (r=",r,") took ",(proc.time() - ptm)[3],"sec."))
  
  # Step 1: Initialization (draw particles from the prior)
  ptm <- proc.time()
  Sample<-matrix(numeric(npart*(4*N_Peaks+3+N_Obs_Cal)),nrow=npart)
  Sample[,scale_G_mask] <- rlnorm(N_Peaks*npart, lPriors$scaG.mu, lPriors$scaG.sd)
  Sample[,scale_L_mask] <- rlnorm(N_Peaks*npart, lPriors$scaL.mu, lPriors$scaL.sd)
  # enforce window and identifiability constrants on peak locations
  for (k in 1:npart) {
    propLoc <- rtruncnorm(N_Peaks, a=min(wl), b=max(wl), mean=lPriors$loc.mu, sd=lPriors$loc.sd)
    Sample[k,location_mask] <- sort(propLoc)
  }
  # optional prior on beta
  if (exists("beta.mu", lPriors) && exists("beta.sd", lPriors)) {
    for (j in scale_G_mask) {
      Sample[,3*N_Peaks+j] <- rtruncnorm(npart, a=0, mean=lPriors$beta.mu[j], sd=lPriors$beta.sd[j])
    }
  } else { # otherwise, use uniform prior
    Sample[,beta_mask] <- runif(N_Peaks*npart, 0, diff(range(spc))/max(conc))
  }
  
  Offset_1<-4*N_Peaks
  log_likelihood_mask <- Offset_1+2
  weight_mask <- Offset_1+1
  
  Offset_2<-Offset_1 + N_Obs_Cal + 1
  Cal_I <- 1
  Sample[,Offset_2+1] <- 1/rgamma(npart, lPriors$noise.nu/2, lPriors$noise.SS/2)
  #Sample[,Offset_1+4] <- 1/rgamma(npart, lPriors$bl.nu/2, lPriors$bl.SS/2)
  Sample[,Offset_2+2] <- Sample[,Offset_2+1]/lPriors$bl.smooth
  print(paste("Mean noise parameter sigma is now",mean(sqrt(Sample[,Offset_2+1]))))
  print(paste("Mean spline penalty lambda is now",mean(Sample[,Offset_2+1]/Sample[,Offset_2+2])))
  # compute log-likelihood:
  g0_Cal <- N_WN_Cal * lPriors$bl.smooth * Pre_Cal
  gi_Cal <- XtX + g0_Cal
  a0_Cal <- lPriors$noise.nu/2
  ai_Cal <- a0_Cal + N_WN_Cal/2
  b0_Cal <- lPriors$noise.SS/2
  for(k in 1:npart) {
    Sigi <- conc[Cal_I] * mixedVoigt(Sample[k,2*N_Peaks+(scale_G_mask)], Sample[k,(scale_G_mask)],
                                     Sample[k,N_Peaks+(scale_G_mask)], Sample[k,3*N_Peaks+(scale_G_mask)], wl)
    Obsi <- spc[Cal_I,] - Sigi
    lambda <- lPriors$bl.smooth # fixed smoothing penalty
    L_Ev <- computeLogLikelihood(Obsi, lambda, lPriors$noise.nu, lPriors$noise.SS,
                                 X_Cal, Rsvd$d, lPriors$bl.precision, lPriors$bl.XtX,
                                 lPriors$bl.orthog, lPriors$bl.Ru)
    Sample[k,log_likelihood_mask]<-L_Ev
  }
  Sample[,weight_mask]<-rep(1/npart,npart)
  T_Sample<-Sample
  T_Sample[,scale_G_mask]<-log(T_Sample[,scale_G_mask]) # scaG
  T_Sample[,scale_L_mask]<-log(T_Sample[,scale_L_mask]) # scaL
  T_Sample[,beta_mask]<-log(T_Sample[,beta_mask]) # amp/beta
  iTime <- proc.time() - ptm
  
  ESS<-1/sum(Sample[,weight_mask]^2)
  MC_Steps<-numeric(1000)
  MC_AR<-numeric(1000)
  ESS_Hist<-numeric(1000)
  ESS_AR<-numeric(1000)
  Kappa_Hist<-numeric(1000)
  Time_Hist<-numeric(1000)
  
  MC_Steps[1]<-0
  MC_AR[1]<-1
  ESS_Hist[1]<-ESS
  ESS_AR[1]<-npart
  Kappa_Hist[1]<-0
  Time_Hist[1]<-iTime[3]
  print(paste("Step 1: initialization for",N_Peaks,"Voigt peaks took",iTime[3],"sec."))
  print(colMeans(Sample[,beta_mask]))
  
  i<-1
  Cal_I <- 1
  MADs<-numeric(4*N_Peaks)
  Alpha<-rate
  MC_AR[1]<-mcAR
  MCMC_MP<-1
  
  Kappa <- 0
  
  # Evidence
  log_evidence <- 0
  
  # End timing init
  init_end_time <- Sys.time()
  init_time_taken <- init_end_time - init_start_time
  
  print(paste0("Took ", init_time_taken, " seconds to init."))
  
  #par(mfrow=c(2,6))
  #plot(density(Sample[,location_mask[1]]), main=paste0("Iteration: ", i))
  repeat{
    #if (i %% 2 == 0) {
    #  plot(density(Sample[,location_mask[1]]), main=paste0("Iteration: ", i)) 
    #}
    
    i<-i+1
    
    
    ptm <- proc.time()
    
    # Determine gamma_{t+1}
    new_Kappa <- calculate_new_gamma(Kappa, Sample, log_likelihood_mask, npart, weight_mask)
    
    # Reweighting
    reweight_res <- reweight_particles(Sample, weight_mask, log_likelihood_mask, new_Kappa, Kappa, log_evidence, Cal_I)
    Temp_ESS <- reweight_res$Temp_ESS
    Sample <- reweight_res$Sample
    log_evidence <- reweight_res$log_evidence
    
    # Update records
    Kappa_Hist[i] <- Kappa
    ESS_Hist[i] <- Temp_ESS
    
    print(paste0("Reweighting took ",(proc.time()-ptm)[3],"sec. for ESS ",Temp_ESS," with new kappa ",Kappa,"."))
    
    
    # Parallel resampling
    ptm <- proc.time()
    
    #if (i > 10) {
    #  # Simple multinomial resampling
    #  resample_res <- resample_particles(npart, Sample, T_Sample, weight_mask)
    #  Sample <- resample_res$Sample
    #  T_Sample <- resample_res$T_Sample
    #} else {
      # Parallel resampling
      idx <- metropolisParallelResampling(Sample[,weight_mask], Sample, T_Sample)  
      
      #Sample <- Sample[idx,]
      #T_Sample <- T_Sample[idx,]
      
      print(paste("*** Resampling with",length(unique(T_Sample[,1])),"unique indices took",(proc.time()-ptm)[3],"sec ***"))
    #}
    
    Sample[,weight_mask] <- rep(1,npart)/npart
    T_Sample[,weight_mask] <- rep(1,npart)/npart

    # Move particles
    move_particles_res <- move_particles(Sample, T_Sample, N_Peaks, npart, weight_mask, MC_Steps, i, 
                                         spc, Cal_I, Kappa_Hist, conc, wl, lPriors, ESS_AR, prev_accept_rate, c_c, number_of_threads)
    MC_Steps <- move_particles_res$MC_Steps
    ESS_AR <- move_particles_res$ESS_AR
    Temp_ESS <- move_particles_res$Temp_ESS
    Acc <- move_particles_res$Acc
    Temp_ESS <- move_particles_res$Temp_ESS
    
    MC_AR[i] <- Acc/(npart*MC_Steps[i])
    
    if (!is.na(destDir) && file.exists(destDir)) {
      iFile<-paste0(destDir,"/Iteration_",i,"/")
      dir.create(iFile)
      save(Sample,file=paste0(iFile,"Sample.rda"))
      print(paste("Interim results saved to",iFile))
    }
    
    print(colMeans(Sample[,beta_mask]))
    print(paste0("Iteration ",i,": MCMC loops (acceptance rate ",MC_AR[i],")"))
    if (Kappa >= 1 || MC_AR[i] < 1/npart) {
      break
    }
    # Update Kappa
    Kappa = new_Kappa
  }
  
  if (Kappa < 1 && MC_AR[i] < 1/npart) {
    print(paste("SMC collapsed due to MH acceptance rate",
                Acc,"/",(npart*MC_Steps[i]),"=", MC_AR[i]))
  }
  
  return(list(priors=lPriors, ess=ESS_Hist[1:i], weights=Sample[,weight_mask], kappa=Kappa_Hist[1:i],
              accept=MC_AR[1:i], mhSteps=MC_Steps[1:i], essAR=ESS_AR[1:i], times=Time_Hist[1:i],
              scale_G=Sample[,scale_G_mask], scale_L=Sample[,scale_L_mask],
              location=Sample[,location_mask], beta=Sample[,beta_mask],
              sigma=sqrt(Sample[,Offset_2+1]), lambda=Sample[,Offset_2+1]/Sample[,Offset_2+2],
              log_evid = log_evidence))
}

move_particles <- function(Sample, T_Sample, N_Peaks, npart, weight_mask, MC_Steps, i, 
                           spc, Cal_I, Kappa_Hist, conc, wl, lPriors, ESS_AR, prev_accept_rate, c_c, number_of_threads) {
  
  Prop_Info <- cov.wt(T_Sample[,1:(4*N_Peaks)], wt=Sample[,weight_mask])
  
  
  possible_scaling_factors <- c(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25)
  suggested_repeats <- rep(0, length(possible_scaling_factors))
  ratios <- rep(0, length(possible_scaling_factors))
  # Do 1 iteration to adaptively chose MCMC repeats
  MC_Steps[i] <- MC_Steps[i] + 1
  
  particles_vec <- 1:npart
  step_space <- npart / length(possible_scaling_factors)
  split_up <- lapply(seq(1,length(particles_vec), step_space),function(i) particles_vec[i:min(i+step_space-1,length(particles_vec))])
  Acc <- 0
  sf <- 1
  for (split in split_up) {
    scaling_factor_selected <- possible_scaling_factors[sf]
    mhCov <- Prop_Info$cov * scaling_factor_selected
    mhChol <- t(chol(mhCov, pivot = FALSE))
    mh_acc <- mhUpdateVoigt(spc, Cal_I, Kappa_Hist[i], conc, wl, Sample[split,], T_Sample[split,], mhChol, lPriors, number_of_threads)
    Acc <- Acc + mh_acc
    prelim_accept_rate <- mh_acc / step_space
    C <- ifelse(prev_accept_rate == 0, 100, ceiling(log(c_c)/log(1-prelim_accept_rate))) 
    suggested_repeats[sf] <- C
    ratios[sf] <- C / scaling_factor_selected
    sf <- sf + 1
  }
  print(suggested_repeats)
  locMin <- which.min(ratios)
  C <- suggested_repeats[locMin]
  if (C < 1) {
    C <- 10
  }
  scaling_factor <- possible_scaling_factors[locMin]
  mhCov <- Prop_Info$cov * scaling_factor
  mhChol <- t(chol(mhCov, pivot = FALSE))
  print(paste("MCMC repeats is ", C, "and scaling factor is ", scaling_factor))
  
  Temp_ESS <- 1/sum(Sample[,weight_mask]^2)
  ESS_AR[i] <- Temp_ESS
  
  for(mcr in 2:C) {
    MC_Steps[i] <- MC_Steps[i] + 1
    mh_acc <- mhUpdateVoigt(spc, Cal_I, Kappa_Hist[i], conc, wl, Sample, T_Sample, mhChol, lPriors, number_of_threads)
    Acc <- Acc + mh_acc
    
    Temp_ESS <- 1/sum(Sample[,weight_mask]^2)
    print(paste(mh_acc,"M-H proposals accepted."))
    ESS_AR[i] <- Temp_ESS
  }
  
  return(list(
    MC_Steps = MC_Steps,
    ESS_AR = ESS_AR,
    Temp_ESS = Temp_ESS, 
    Acc = Acc,
    Temp_ESS = Temp_ESS
  ))
}

# Reweight
reweight_particles <- function(Sample, weight_mask, log_likelihood_mask, new_Kappa, Kappa, log_evidence, Cal_I) {
  log_weights <- log(Sample[,weight_mask]*exp((new_Kappa-Kappa)*(Sample[,weight_mask+Cal_I]-max(Sample[,weight_mask+Cal_I]))))
  #log_weights <- log(Sample[,weight_mask]) + (new_Kappa - Kappa) * 
  #  Sample[,log_likelihood_mask]
  
  ## Log evidence
  log_evidence <- log_evidence + logSumExp(log_weights)
  
  # Numerically stabilise before exponentiating
  log_weights <- log_weights - max(log_weights)
  Sample[,weight_mask] <- exp(log_weights)
  
  # Normalise
  Sample[,weight_mask] <- Sample[,weight_mask] / sum(Sample[,weight_mask])
  
  # Return the ESS
  Temp_ESS <- 1/sum(Sample[,weight_mask]^2)
  
  return(list(log_evidence=log_evidence, Sample=Sample, Temp_ESS=Temp_ESS))
}

# Resampling 
resample_particles <- function(npart, Sample, T_Sample, weight_mask) {
  ptm <- proc.time()
  
  ReSam<-sample(1:npart, size=npart, replace=T, prob=Sample[,weight_mask])
  Sample<-Sample[ReSam,]
  T_Sample<-T_Sample[ReSam,]
  
  Sample[,weight_mask]<-rep(1/npart,npart)
  T_Sample[,weight_mask]<-rep(1/npart,npart)
  
  print(paste("*** Resampling with",length(unique(T_Sample[,1])),"unique indices took",(proc.time()-ptm)[3],"sec ***"))
  
  return(list(Sample = Sample, T_Sample = T_Sample))
}

# Calculate new gamma
calculate_new_gamma <- function(Kappa, Sample, log_likelihood_mask, npart, weight_mask) {
  ess1 <- calc_ESS(1, Kappa, Sample[,weight_mask], Sample[,log_likelihood_mask])
  
  # Bisection method to maintain ESS at N/2
  if (ess1 > npart/2) {
    new_Kappa = 1
  } else {
    bi_section = uniroot(
      function(new_Kappa) calc_ESS(new_Kappa, Kappa, Sample[,weight_mask], Sample[,log_likelihood_mask]) 
      - npart/2,
      lower = Kappa, 
      upper = 1
    )
    new_Kappa <- bi_section$root
  }
  print(new_Kappa)
  return(new_Kappa)
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
