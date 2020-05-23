library(doMC)
library(serrsBayes)
library(matrixStats)
library(Rcpp)
source("/home/lachlan/Honours/serrsBayes/R/fitVoigtPeaksSMC_update.R")

set.seed(1234)

runs <- 1:10
log_evid_rec <- c()
accepts_rec <- c()

#for (i in runs) {

data("methanol", package = "serrsBayes")
wavenumbers <- methanol$wavenumbers
spectra <- methanol$spectra

peakLocations <- c(1033, 1106, 1149, 1448)
nPK <- length(peakLocations)
pkIdx <- numeric(nPK)
for (i in 1:nPK) {
  pkIdx[i] <- which.min(wavenumbers < peakLocations[i])
}
nWL <- length(wavenumbers)
plot(wavenumbers, spectra[1,], type='l', col=4,
     xlab=expression(paste("Raman shift (cm"^{-1}, ")")), ylab="Intensity (a.u.)", main="Observed Raman spectrum for methanol")
points(peakLocations, spectra[1,pkIdx], cex=2, col=2)
text(peakLocations + c(100,20,40,0), spectra[1,pkIdx] + c(0,700,400,700), labels=peakLocations)
lPriors2 <- list(loc.mu=peakLocations, loc.sd=rep(50,nPK), scaG.mu=log(16.47) - (0.34^2)/2,
                 scaG.sd=0.34, scaL.mu=log(25.27) - (0.4^2)/2, scaL.sd=0.4, noise.nu=5,
                 noise.sd=50, bl.smooth=1, bl.knots=50)

wl <- wavenumbers
spc <- spectra
lPriors <- lPriors2
npart <- 3000
rate=0.9
mcAR=0.23
mcSteps=10
minESS=npart/2
destDir=NA
conc=rep(1.0,nrow(spc))

sourceCpp("/home/lachlan/Honours/serrsBayes/src/mixVoigt.cpp", verbose = FALSE, showOutput = FALSE)

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

i<-i+1


ptm <- proc.time()

# Determine gamma_{t+1}
new_Kappa <- calculate_new_gamma(Kappa, Sample, log_likelihood_mask, npart, weight_mask)

# Reweighting
reweight_res <- reweight_particles(Sample, weight_mask, log_likelihood_mask, new_Kappa, Kappa, log_evidence)
Temp_ESS <- reweight_res$Temp_ESS
Sample <- reweight_res$Sample
log_evidence <- reweight_res$log_evidence

# Update records
Kappa_Hist[i] <- Kappa
ESS_Hist[i] <- Temp_ESS

print(paste0("Reweighting took ",(proc.time()-ptm)[3],"sec. for ESS ",Temp_ESS," with new kappa ",Kappa,"."))


#resampleParticlesNew(Sample[,weight_mask], Sample, runif(npart * 10), sample(1:npart, npart * 10, replace = TRUE))


#idx <- resampleParticles(log(Sample[,weight_mask]), Sample)
#idx
