#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>
#include <set>

using namespace std;
using namespace Eigen;

double calculateNewKappa(double kappa, ArrayXd weights, ArrayXd logLike, int nPart);
double calcESS(double newGamma, double oldGamma, ArrayXd weights, ArrayXd logLike);
double bisection(double a, double b, function<double (double updateKappa)> func);
double reweightParticles(double newKappa, double kappa, Eigen::Ref<Eigen::ArrayXd> weights, ArrayXd logLike, int nPart);
ArrayXi metropolisResampling(ArrayXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample, int weightMask);
void moveParticles(VectorXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample, int nPeaks, double kappa);
long mhUpdateVoigt(Eigen::VectorXd spectra, unsigned n, double kappa, Eigen::VectorXd conc, Eigen::VectorXd wavenum,
                   Eigen::Ref<Eigen::MatrixXd> thetaMx, Eigen::Ref<Eigen::MatrixXd> logThetaMx, Eigen::MatrixXd mhChol);

// Helper functions
ArrayXd toScalarArray(double scalar, int n);
void swapEles(int & a, int & b);
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
void writeToCSVfile(string name, MatrixXd matrix);

// Defines
#define PI 3.1415

template<typename M>
M loadCsvMatrix(const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

// Import a lot of stuff to setup
int nPeaks = 4;
int nPart = 4000;

// Setup masks
auto scaleGmask = seq(0, nPeaks-1);
auto scaleLmask = seq(nPeaks, (2*nPeaks-1));
auto locationMask = seq((2*nPeaks), (3*nPeaks-1));
auto betaMask = seq((3*nPeaks), (4*nPeaks-1));
int offset1 = 4*nPeaks;
int logLikelihoodMask = offset1+1;
int weightMask = offset1;

// Load in matrices
MatrixXd sample = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/sample.csv");
MatrixXd tSample = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/tSample.csv");

// Priors
MatrixXd basisMx = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blBasis.csv");
MatrixXd precMx = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blPrecision.csv");
MatrixXd xTx = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blXtX.csv");
MatrixXd aMx = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blOrthog.csv");
MatrixXd ruMx = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blRu.csv");
VectorXd eigVal = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/blEigen.csv").col(0);

// Spectra
VectorXd spectra = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/spectra.csv").row(0);
VectorXd wl = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/wl.csv").col(0);

int main() { 
    // Playground

    // Some easy to port setup
    int i = 1;
    double kappa = 0;
    double newKappa;

    do {
        // Increment iteration count
        i += 1;

        cout << "Min: " << sample(all, locationMask).col(0).array().minCoeff() << endl;
        cout << "Mean: " << sample(all, locationMask).col(0).array().mean() << endl;
        cout << "Max: " << sample(all, locationMask).col(0).array().maxCoeff() << endl;

        // Determine gamma_{t+1}
        newKappa = calculateNewKappa(kappa, sample(all, weightMask), sample(all, logLikelihoodMask), nPart);
        cout << "New Kappa: " << newKappa << endl;

        // Reweight particles
        double tempESS = reweightParticles(newKappa, kappa, sample(all, weightMask), sample(all, logLikelihoodMask), nPart);
        cout << "Temp ESS: " << tempESS << endl;

        // Resampling
        ArrayXi idx = metropolisResampling(sample(all, weightMask), sample, tSample, weightMask);
        std::set<int> q{idx.begin(), idx.end()}; // get a count for how many unique
        cout << "Resampled unique count: " << q.size() << endl;

        // Move particles
        moveParticles(sample(all, weightMask), sample, tSample, nPeaks, kappa);
        
        // Update Kappa
        kappa = newKappa;

    } while (kappa < 1);

    // Write the results to disk
    writeToCSVfile("sample.csv", sample);
}

void writeToCSVfile(string name, MatrixXd matrix) {
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
 }

void moveParticles(VectorXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample, int nPeaks, double kappa) {
  MatrixXd covMat = tSample(all, seq(0, 4*nPeaks - 1));
  MatrixXd centered = covMat.rowwise() - weights.transpose() * covMat;
  MatrixXd covWt = (centered.adjoint() * centered) / double(covMat.rows() - 1);
  //cout << covWt << endl;

  MatrixXd mhCov = covWt * MatrixXd::Identity(covWt.rows(), covWt.cols()) * 0.1;
  LLT<MatrixXd> lltOfCov(mhCov);
  MatrixXd mhChol = lltOfCov.matrixL();
  //cout << mhChol << endl;

  VectorXd conc(1);
  conc(0) = 1;

  for (int mcr = 0; mcr < 5; mcr++) {
    long mhAcc = mhUpdateVoigt(spectra, 1, kappa, conc, wl, sample, tSample, mhChol);
    cout << mhAcc << " moves accepted." << endl;
  }

}

double computeLogLikelihood(Eigen::VectorXd obsi, double lambda, double prErrNu, double prErrSS,
            Eigen::MatrixXd basisMx, Eigen::VectorXd eigVal, Eigen::MatrixXd precMx,
            Eigen::MatrixXd xTx, Eigen::MatrixXd aMx, Eigen::MatrixXd ruMx)
{
  double nWL = obsi.size();
  double a0_Cal = prErrNu/2.0;
  double ai_Cal = a0_Cal + nWL/2.0;

  MatrixXd g0_Cal = precMx * nWL * lambda;
  MatrixXd g0LU(g0_Cal);
  MatrixXd gi_Cal = xTx + g0_Cal;
  MatrixXd giLU(gi_Cal);
  double g0_Det = log(abs(g0LU.determinant()));
  double gi_Det = log(abs(giLU.determinant()));
//  Rcpp::Rcout << g0_Det << "; " << gi_Det << "; ";
  VectorXd b = aMx.transpose() * obsi;
  if (!b.allFinite()) b = aMx.transpose() * obsi.transpose(); // row vector
//  Rcpp::Rcout << b.allFinite() << "; " << b.mean() << "; ";
  ArrayXd ePlus = eigVal.array() * lambda * nWL + 1.0;
  VectorXd bRatio = (b.array() / ePlus).matrix();
  VectorXd mi_New = ruMx * bRatio;
  double sqDiff = obsi.squaredNorm() - mi_New.transpose() * gi_Cal * mi_New;
//  Rcpp::Rcout << obsi.squaredNorm() << "; " << mi_New.transpose() * gi_Cal * mi_New << "; ";
  double bi_Cal = (prErrSS + sqDiff)/2.0;
//  Rcpp::Rcout << sqDiff << "; " << bi_Cal << "; ";

  // log-likelihood:
  double L_Ev = -((nWL/2.0)*log(2.0*PI)) + 0.5*g0_Det - 0.5*gi_Det;
  L_Ev +=  a0_Cal*log(prErrSS) - ai_Cal*log(bi_Cal) + lgamma(ai_Cal) - lgamma(a0_Cal);
  return L_Ev;
}

double sumDnorm(Eigen::VectorXd x, Eigen::VectorXd mean, Eigen::VectorXd sd)
{
  double logLik = 0;
  for (int pk=0; pk < x.size(); pk++)
  {
    logLik += -pow(x[pk] - mean[pk], 2.0)/(2*pow(sd[pk],2.0)) - log(sd[pk] * sqrt(2*PI));
  }
  return logLik;
}

double sumDlogNorm(Eigen::VectorXd x, double meanlog, double sdlog)
{
  double var = pow(sdlog, 2.0);
  ArrayXd sqDiff = (x.array().log() - meanlog).pow(2.0);
  ArrayXd logConst = (x.array() * sdlog * sqrt(2*PI)).log();
  ArrayXd logLik = -sqDiff/(2*var) - logConst;
  return logLik.sum();
}

Eigen::VectorXd dNorm(Eigen::VectorXd Cal_V, double loc, double sd)
{
  VectorXd Sigi = VectorXd::Zero(Cal_V.size());
  for (int i=0; i < Cal_V.size(); i++)
  {
    Sigi[i] = 1/(sqrt(2*PI)*sd) * exp(-pow(Cal_V[i] - loc, 2)/(2*pow(sd,2)));
  }
  return Sigi;
}

Eigen::VectorXd dCauchy(Eigen::VectorXd Cal_V, double loc, double scale)
{
  VectorXd Sigi = VectorXd::Zero(Cal_V.size());
  for (int i=0; i < Cal_V.size(); i++)
  {
    Sigi[i] = 1/(PI*scale*(1 + pow((Cal_V[i] - loc)/scale, 2)));
  }
  return Sigi;
}

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

long mhUpdateVoigt(Eigen::VectorXd spectra, unsigned n, double kappa, Eigen::VectorXd conc, Eigen::VectorXd wavenum,
                   Eigen::Ref<Eigen::MatrixXd> thetaMx, Eigen::Ref<Eigen::MatrixXd> logThetaMx, Eigen::MatrixXd mhChol)
{
  // priors
  double prErrNu = 5;
  double prErrSS = 12500;
  double prScaGmu = 2.743741; // squared exponential (Gaussian) peaks
  double prScaGsd = 0.34;
  double prScaLmu = 3.149618; // Lorentzian (Cauchy) peaks
  double prScaLsd = 0.4;
  VectorXd prLocMu(4);
  prLocMu << 1033, 1106, 1149, 1448;
  VectorXd prLocSD(4);
  prLocSD << 50, 50, 50, 50;
  double lambda = 1;
  int nPK = prLocMu.size();
  int nPart = thetaMx.rows();
  int nWL = wavenum.size();

  // RNG is not thread-safe
  std::default_random_engine generator;
  std::uniform_real_distribution<double> sampleUniform(0.0, 1.0);
  std::normal_distribution<double> sampleStandardNormal{0,1};

  VectorXd rUnif(nPart);
  for (int i = 0; i < nPart; i++) rUnif(i) = sampleUniform(generator);
  VectorXd stdNorm(nPK * nPart * 4);
  for (int i = 0; i < (nPK * nPart * 4); i++) stdNorm(i) = sampleStandardNormal(generator);
  

  long accept = 0;
  
  #pragma omp parallel for reduction(+:accept)
  for (int pt = 0; pt < nPart; pt++)
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
    std::sort(Prop_Theta.data() + 2*nPK,
              Prop_Theta.data() + 3*nPK - 1); // for identifiability

    VectorXd sigi = conc(n-1) * mixedVoigt(Prop_Theta.segment(2*nPK,nPK), Prop_Theta.segment(0,nPK),
       Prop_Theta.segment(nPK,nPK), Prop_Theta.segment(3*nPK,nPK), wavenum);
    VectorXd obsi = spectra - sigi;

    // smoothing spline:
    //double lambda = thetaMx(pt,4*nPK+2) / thetaMx(pt,4*nPK+3);
    // log-likelihood:
    double L_Ev = computeLogLikelihood(obsi, lambda, prErrNu, prErrSS, basisMx, eigVal,
                                       precMx, xTx, aMx, ruMx);
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
      oldLogLik(i) = computeLogLikelihood(obsi, lambda, prErrNu, prErrSS, basisMx, eigVal,
                precMx, xTx, aMx, ruMx);
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
      accept += 1;
    }
  }
  return accept;
}

ArrayXi metropolisResampling(ArrayXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample, int weightMask) {
  // Initialise return result
  const int sizeWeights = weights.size();
  
  // The higher, the better (less bias)
  const int B = 500;
  ArrayXi idx(sizeWeights);
  
  // Rng
  std::default_random_engine generator;
  std::uniform_int_distribution<int> sampleRandomInt(0, sizeWeights - 1);
  std::uniform_real_distribution<double> sampleUniform(0.0, 1.0);

  for (int i = 0; i < sizeWeights; i++) {
    int k = i;
    for (int n = 0; n < B; n++) {
      double u = sampleUniform(generator);
      int j = sampleRandomInt(generator);
      if (u < (weights[j] / weights[k])) {
        k = j;
      }
    }
    idx[i] = k;
  }
  
  
  //Rcpp::Rcout << "Max of idx is " << max(idx) << "\n";
  
  // permute the index vector to ensure Condition 9 of Murray, Lee & Jacob (2015)
  for (int i = 0; i < sizeWeights; i++)
  {
    if ((idx[i] != i) && (idx[idx[i]] != idx[i]))
    {
      swapEles(idx[i], idx[idx[i]]);
      i = i - 1;
    }
  }
  
  for (int p = 0; p < sizeWeights; p++)
  {
    // do nothing unless the particle has no offspring
    if (idx[p] != p)
    {
      for (int j=0; j < sample.cols(); j++)
      {
        sample(p, j) = sample(idx[p], j);
        tSample(p, j) = tSample(idx[p], j);
      }
    }
  }

  sample(all,weightMask) = toScalarArray(1/(double)sizeWeights, sizeWeights);
  tSample(all,weightMask) = toScalarArray(1/(double)sizeWeights, sizeWeights);

  // Return the resample ids
  return idx;
}

// Swap elements in array
void swapEles(int & a, int & b) {
  int temp = a;
  a = b;
  b = temp;
}

double reweightParticles(double newKappa, double kappa, Eigen::Ref<Eigen::ArrayXd> weights, ArrayXd logLike, int nPart) {
    ArrayXd maxLogLike = toScalarArray(logLike.maxCoeff(), logLike.size());
    
    // Reweight
    ArrayXd logWeights = (weights * exp((newKappa - kappa) * (logLike - maxLogLike))).log();

    // TODO: Log evidence calculations

    // Numerically stabilise before exponentiating
    ArrayXd maxLogWeight = toScalarArray(logWeights.maxCoeff(), logWeights.size());
    logWeights = logWeights - maxLogWeight;

    // Normalise
    ArrayXd normalisedWeights = logWeights.exp();
    normalisedWeights = normalisedWeights / normalisedWeights.sum();

    // Update the weights 
    weights = normalisedWeights;

    // Return the ESS
    double tempESS = 1 / (weights.square().sum());
    return(tempESS);
}

double calculateNewKappa(double kappa, ArrayXd weights, ArrayXd logLike, int nPart) {
    // Define the new kappa
    double newKappa;

    double ess1 = calcESS(1, kappa, weights, logLike);
    
    // Bisection method to maintain ESS at N/2
    if (ess1 > nPart/2) {
        newKappa = 1;
    } else {
        function<double (double updateKappa)> f = [&](double updateKappa) { 
            return (
                calcESS(updateKappa, kappa, weights, logLike) - nPart / 2
            );
        };
        newKappa = bisection(kappa, 1, f);
    }
    return newKappa;
}

double bisection(double a, double b, function<double (double updateKappa)> func) {
    if (func(a) * func(b) >= 0) {
        cout << "Incorrect a and b" << endl;
        return -1;
    }
 
    double c = a;
 
    while ((b-a) >= 0.0001) {
        c = (a+b)/2;
        if (func(c) == 0.0){
            return c;
        } else if (func(c)*func(a) < 0){
            b = c;
        } else {
            a = c;
        }
    }
    return c;
}

ArrayXd toScalarArray(double scalar, int n) {
    return scalar * ArrayXd::Ones(n);
}

double calcESS(double newGamma, double oldGamma, ArrayXd weights, ArrayXd logLike) {
    int n = weights.size();
    // By notes
    ArrayXd logWeights = weights.log() + toScalarArray(newGamma - oldGamma, n) * logLike;

    // Numerically stabilise before exponentiating
    logWeights = logWeights - toScalarArray(logWeights.maxCoeff(), n);
    ArrayXd newWeights = logWeights.exp();

    // Normalise
    newWeights = newWeights / newWeights.sum();

    // Return the ESS
    double ESS = 1 / newWeights.square().sum();
    return(ESS);

}