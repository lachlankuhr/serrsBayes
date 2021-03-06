#install.packages("serrsBayes", repo="https://cran.ms.unimelb.edu.au/")
library(serrsBayes)
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
data("result2", package = "serrsBayes")
tm2 <- system.time(result2 <- fitVoigtPeaksSMC(wavenumbers, spectra, lPriors2, npart=3000))
result2$time <- tm2

print(paste(result2$time["elapsed"]/60, "minutes"))
samp.idx <- 1:nrow(result2$location)
plot(wavenumbers, spectra[1,], type='l', lwd=3,
     xlab=expression(paste("Raman shift (cm"^{-1}, ")")), ylab="Intensity (a.u.)", main="Fitted model with Voigt peaks")
samp.mat <- resid.mat <- matrix(0,nrow=length(samp.idx), ncol=nWL)
samp.sigi <- samp.lambda <- numeric(length=nrow(samp.mat))
for (pt in 1:length(samp.idx)) {
  k <- samp.idx[pt]
  samp.mat[pt,] <- mixedVoigt(result2$location[k,], result2$scale_G[k,],
                              result2$scale_L[k,], result2$beta[k,], wavenumbers)
  samp.sigi[pt] <- result2$sigma[k]
  samp.lambda[pt] <- result2$lambda[k]
  
  Obsi <- spectra[1,] - samp.mat[pt,]
  g0_Cal <- length(Obsi) * samp.lambda[pt] * result2$priors$bl.precision
  gi_Cal <- crossprod(result2$priors$bl.basis) + g0_Cal
  mi_Cal <- as.vector(solve(gi_Cal, crossprod(result2$priors$bl.basis, Obsi)))
  
  bl.est <- result2$priors$bl.basis %*% mi_Cal # smoothed residuals = estimated basline
  lines(wavenumbers, bl.est, col="#C3000009")
  lines(wavenumbers, bl.est + samp.mat[pt,], col="#00C30009")
  resid.mat[pt,] <- Obsi - bl.est[,1]
}