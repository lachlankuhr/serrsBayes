#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>

using namespace std;
using namespace Eigen;

double calculateNewKappa(double kappa, ArrayXd weights, ArrayXd logLike, int nPart);
double calcESS(double newGamma, double oldGamma, ArrayXd weights, ArrayXd logLike);
double bisection(double a, double b, function<double (double updateKappa)> func);
double reweightParticles(double newKappa, double kappa, Eigen::Ref<Eigen::ArrayXd> weights, ArrayXd logLike, int nPart);
ArrayXi metropolisResampling(ArrayXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample);

// Helper functions
ArrayXd toScalarArray(double scalar, int n);
void swapEles(int & a, int & b);

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

int main() { 
    // Playground

    // Some easy to port setup
    int i = 1;
    double kappa = 0;
    double newKappa;

    // Load in matrices
    MatrixXd sample = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/sample.csv");
    MatrixXd tSample = loadCsvMatrix<MatrixXd>("/home/lachlan/Honours/serrsBayes/src/data/tSample.csv");

    do {
        // Increment iteration count
        i += 1;

        // Determine gamma_{t+1}
        newKappa = calculateNewKappa(kappa, sample(all, weightMask), sample(all, logLikelihoodMask), nPart);
        cout << "New Kappa: " << newKappa << endl;

        // Reweight particles
        double tempESS = reweightParticles(newKappa, kappa, sample(all, weightMask), sample(all, logLikelihoodMask), nPart);
        cout << "Temp ESS: " << tempESS << endl;

        // Resampling
        ArrayXi idx = metropolisResampling(sample(all, weightMask), sample, tSample);
        cout << idx << endl;
        while(true);

    } while (true);
}

ArrayXi metropolisResampling(ArrayXd weights, Eigen::Ref<Eigen::MatrixXd> sample, Eigen::Ref<Eigen::MatrixXd> tSample) {
  // Initialise return result
  const int sizeWeights = weights.size();
  
  // The higher, the better (less bias)
  const int B = 6000;
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