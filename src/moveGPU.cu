#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <random>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "device_launch_parameters.h"

using namespace std;
using namespace Eigen;

// Defines
#define PI 3.1415

// priors - NEED TO UPDATE THESE IF THE DATA CHANGES
#define prErrNu 5
#define prErrSS 12500
#define prScaGmu 2.743741 // squared exponential (Gaussian) peaks
#define prScaGsd 0.34
#define prScaLmu 3.149618 // Lorentzian (Cauchy) peaks
#define prScaLsd 0.4
#define lambda 1

__device__
double computeLogLikelihood(Eigen::VectorXd obsi, Eigen::MatrixXd basisMx, Eigen::VectorXd eigVal, Eigen::MatrixXd precMx,
            Eigen::MatrixXd xTx, Eigen::MatrixXd aMx, Eigen::MatrixXd ruMx,
            double g0_Det, double gi_Det)
{
  double nWL = obsi.size();
  double a0_Cal = prErrNu/2.0;
  double ai_Cal = a0_Cal + nWL/2.0;

  MatrixXd g0_Cal = precMx * nWL * lambda;
  MatrixXd g0LU(g0_Cal);
  MatrixXd gi_Cal = xTx + g0_Cal;

  VectorXd b = aMx.transpose() * obsi;
  ArrayXd ePlus = eigVal.array() * lambda * nWL + 1.0;
  VectorXd bRatio = (b.array() / ePlus).matrix();
  VectorXd mi_New = ruMx * bRatio;
  double sqDiff = obsi.squaredNorm() - mi_New.transpose() * gi_Cal * mi_New;
  double bi_Cal = (prErrSS + sqDiff)/2.0;

  // log-likelihood:
  double L_Ev = -((nWL/2.0)*log(2.0*PI)) + 0.5*g0_Det - 0.5*gi_Det;
  L_Ev +=  a0_Cal*log(prErrSS) - ai_Cal*log(bi_Cal) + lgamma(ai_Cal) - lgamma(a0_Cal);
  return L_Ev;
}

__device__
double sumDnorm(Eigen::VectorXd x, Eigen::VectorXd mean, Eigen::VectorXd sd)
{
  double logLik = 0;
  for (int pk=0; pk < x.size(); pk++)
  {
    logLik += -pow(x[pk] - mean[pk], 2.0)/(2*pow(sd[pk],2.0)) - log(sd[pk] * sqrt(2*PI));
  }
  return logLik;
}

__device__
double sumDlogNorm(Eigen::VectorXd x, double meanlog, double sdlog)
{
  double var = pow(sdlog, 2.0);
  ArrayXd sqDiff = (x.array().log() - meanlog).pow(2.0);
  ArrayXd logConst = (x.array() * sdlog * sqrt(2*PI)).log();
  ArrayXd logLik = -sqDiff/(2*var) - logConst;
  return logLik.sum();
}

__device__
Eigen::VectorXd dNorm(Eigen::VectorXd Cal_V, double loc, double sd)
{
  VectorXd Sigi = VectorXd::Zero(Cal_V.size());
  for (int i=0; i < Cal_V.size(); i++)
  {
    Sigi[i] = 1/(sqrt(2*PI)*sd) * exp(-pow(Cal_V[i] - loc, 2)/(2*pow(sd,2)));
  }
  return Sigi;
}

__device__
Eigen::VectorXd dCauchy(Eigen::VectorXd Cal_V, double loc, double scale)
{
  VectorXd Sigi = VectorXd::Zero(Cal_V.size());
  for (int i=0; i < Cal_V.size(); i++)
  {
    Sigi[i] = 1/(PI*scale*(1 + pow((Cal_V[i] - loc)/scale, 2)));
  }
  return Sigi;
}

__device__
double calcVoigtFWHM(double f_G, double f_L)
{
  // combined scale is the average of the scales of the Gaussian/Lorentzian components
  double Temp_d = pow(f_G,5)
    + 2.69269*pow(f_G,4)*f_L
    + 2.42843*pow(f_G,3)*pow(f_L,2)
    + 4.47163*pow(f_G,2)*pow(f_L,3)
    + 0.07842*f_G*pow(f_L,4)
    + pow(f_L,5);
  return pow(Temp_d, 0.2);
}

__device__
Eigen::VectorXd mixedVoigt(Eigen::VectorXd location, Eigen::VectorXd scale_G, Eigen::VectorXd scale_L, Eigen::VectorXd amplitude, Eigen::VectorXd wavenum)
{
  VectorXd Sigi = VectorXd::Zero(wavenum.size());
  for (int j=0; j < location.size(); j++)
  {
    // combined scale is the average of the scales of the Gaussian/Lorentzian components
    double f_G = 2.0*scale_G(j)*sqrt(2.0*PI);
    double f_L = 2.0*scale_L(j);
    double Temp_f = calcVoigtFWHM(f_G, f_L);

    // (0,1) Voigt parameter gives the mixing proportions of the two components
    double Temp_e = 1.36603*(f_L/Temp_f) - 0.47719*pow(f_L/Temp_f, 2) + 0.11116*pow(f_L/Temp_f, 3);

    // weighted additive combination of Cauchy and Gaussian functions
    Sigi += amplitude[j] * (Temp_e*dCauchy(wavenum,location[j],Temp_f/2.0) + (1.0-Temp_e)*dNorm(wavenum,location[j],Temp_f/(2.0*sqrt(2.0*log(2.0)))))/(Temp_e*(1.0/(PI*(Temp_f/2.0))) + (1.0-Temp_e)*(1.0/sqrt(2.0*PI*pow(Temp_f/(2.0*sqrt(2.0*log(2.0))), 2.0))));
  }
  return Sigi;
}  

__device__
Eigen::VectorXd copyLogProposals(int nPK, Eigen::VectorXd T_Prop_Theta)
{
  VectorXd Prop_Theta(4*nPK);
  for (int par = 0; par < 4; par++)
  {
    if (par != 2)
    {
      Prop_Theta.segment(par*nPK,nPK) = T_Prop_Theta.segment(par*nPK,nPK).array().exp();
    }
    else
    {
      Prop_Theta.segment(par*nPK,nPK) = T_Prop_Theta.segment(par*nPK,nPK);
    }
  }
  return Prop_Theta;
}

__global__
void mhUpdateVoigt(double * d_spectra, unsigned n, double kappa, Eigen::VectorXd conc, double * d_wavenum, double * d_thetaMx, double * d_logThetaMx, double * d_mhChol, double g0_Det, double gi_Det, double * d_basisMx, double * d_precMx, double * d_eigVal, double * d_xTx, double * d_aMx, double * d_ruMx, ArrayXd rUnif, ArrayXd stdNorm)
{
  Map<VectorXd> spectra(d_spectra, 331);
  Map<MatrixXd> thetaMx(d_thetaMx, 20, 20);
  Map<MatrixXd> logThetaMx(d_logThetaMx, 20, 20);
  Map<VectorXd> wavenum(d_wavenum, 331);
  Map<MatrixXd> basisMx(d_basisMx, 331, 54);
  Map<MatrixXd> precMx(d_precMx, 54, 54);
  Map<MatrixXd> xTx(d_xTx, 54, 54);
  Map<MatrixXd> aMx(d_aMx, 331, 54);
  Map<MatrixXd> ruMx(d_ruMx, 54, 54);
  Map<MatrixXd> mhChol(d_mhChol, 16, 16);
  Map<VectorXd> eigVal(d_eigVal, 54);

  VectorXd prLocMu(4);
  prLocMu << 1033, 1106, 1149, 1448;
  VectorXd prLocSD(4);
  prLocSD << 50, 50, 50, 50;
  int nPK = prLocMu.size();
  int nPart = thetaMx.rows();
  int nWL = wavenum.size();

  for (int pt = blockIdx.x * blockDim.x + threadIdx.x; pt < nPart; pt += blockDim.x * gridDim.x) 
  {
    VectorXd theta(nPK*4), logTheta(nPK*4), stdVec(nPK*4);
    for (int pk = 0; pk < nPK*4; pk++)
    {
      stdVec(pk) = stdNorm(pt*nPK*4 + pk);
      theta(pk) = thetaMx(pt,pk);
      logTheta(pk) = logThetaMx(pt,pk);
    }
    VectorXd T_Prop_Theta = mhChol * stdVec + logTheta;
    VectorXd Prop_Theta = copyLogProposals(nPK, T_Prop_Theta);
    // enforce the boundary condition for proposed peak locations
    for (int pk = 0; pk < nPK; pk++)
    {
      if (Prop_Theta(2*nPK+pk) < wavenum(0) || Prop_Theta(2*nPK+pk) > wavenum(nWL-1))
      {
        Prop_Theta(2*nPK+pk) = theta(2*nPK+pk);
      }
    }
    thrust::device_ptr<double> propThetaThrustDevice = thrust::device_pointer_cast(&Prop_Theta(0));
    thrust::sort(thrust::device, propThetaThrustDevice + 2*nPK, propThetaThrustDevice + 3*nPK - 1); // for identifiability

    VectorXd sigi = conc(n-1) * mixedVoigt(Prop_Theta.segment(2*nPK,nPK), Prop_Theta.segment(0,nPK),
       Prop_Theta.segment(nPK,nPK), Prop_Theta.segment(3*nPK,nPK), wavenum);
    VectorXd obsi = spectra - sigi;

    // smoothing spline:
    //double lambda = thetaMx(pt,4*nPK+2) / thetaMx(pt,4*nPK+3);
    // log-likelihood:
    double L_Ev = computeLogLikelihood(obsi, basisMx, eigVal,
                                       precMx, xTx, aMx, ruMx, g0_Det, gi_Det);
    double lLik = kappa*L_Ev + sumDlogNorm(Prop_Theta.segment(0,nPK), prScaGmu, prScaGsd);
    lLik += sumDlogNorm(Prop_Theta.segment(nPK,nPK), prScaLmu, prScaLsd);
    lLik += sumDnorm(Prop_Theta.segment(2*nPK,nPK), prLocMu, prLocSD);
    lLik += -kappa*thetaMx(pt,4*nPK+n) - sumDlogNorm(theta.segment(0,nPK), prScaGmu, prScaGsd);
    lLik -= sumDlogNorm(theta.segment(nPK,nPK), prScaLmu, prScaLsd);
    lLik -= sumDnorm(theta.segment(2*nPK,nPK), prLocMu, prLocSD);

    // account for previous observations when n > 1
    VectorXd oldLogLik(n);
    for (int i=0; i < n-1; i++) {
      sigi = conc(i) * mixedVoigt(Prop_Theta.segment(2*nPK,nPK), Prop_Theta.segment(0,nPK),
                  Prop_Theta.segment(nPK,nPK), Prop_Theta.segment(3*nPK,nPK), wavenum);
      obsi = spectra - sigi;
      oldLogLik(i) = computeLogLikelihood(obsi, basisMx, eigVal,
                precMx, xTx, aMx, ruMx, g0_Det, gi_Det);
      lLik += oldLogLik(i);
      lLik -= thetaMx(pt,4*nPK+i+1);
    }
    oldLogLik(n-1) = L_Ev;

    if (std::isfinite(lLik) && log(rUnif(pt)) < lLik)
    {
      for (int pk=0; pk < nPK*4; pk++)
      {
         logThetaMx(pt,pk) = T_Prop_Theta(pk);
         thetaMx(pt,pk) = Prop_Theta(pk);
      }
      for (int i=0; i < n; i++) {
        logThetaMx(pt,4*nPK+i+1) = oldLogLik(i);
        thetaMx(pt,4*nPK+i+1) = oldLogLik(i);
      }
    }
  }
}

__host__
void jiggleParticles(double * d_spectra, unsigned n, double kappa, Eigen::VectorXd conc, double * d_wavenum, double * d_mhChol, double g0_Det, double gi_Det, double * d_basisMx, double * d_precMx, double * d_eigVal, double * d_xTx, double * d_aMx, double * d_ruMx, ArrayXd rUnif, ArrayXd stdNorm, double * d_sample, double * d_tSample, int grid_size, int block_size) {
    cout << "Jiggle." << std::flush;

    for (int mcr = 0; mcr < 5; mcr++) {
      mhUpdateVoigt<<<block_size, grid_size>>>(d_spectra, 1, kappa, conc, d_wavenum, d_sample, d_tSample, d_mhChol, g0_Det, gi_Det, d_basisMx, d_precMx, d_eigVal, d_xTx, d_aMx, d_ruMx, rUnif, stdNorm);
      cudaDeviceSynchronize();
    }
    cout << "Finished jiggle." << std::flush;
}