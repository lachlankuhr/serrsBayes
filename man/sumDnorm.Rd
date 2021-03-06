% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{sumDnorm}
\alias{sumDnorm}
\title{Sum log-likelihoods of Gaussian.}
\usage{
sumDnorm(x, mean, sd)
}
\arguments{
\item{x}{Vector of i.i.d. Gaussian random varibles}

\item{mean}{Vector of means}

\item{sd}{Vector of standard deviations}
}
\value{
log-likelihood of x
}
\description{
This is an internal function that is only exposed on the public API for unit testing purposes.
}
\details{
The sum of the log-likelihoods (log of the product of the likelihoods)
for independent, identically-distributed, Gaussian random variables.
Note: this Rcpp function is thread-safe, unlike the equivalent alternatives.
}
\examples{
  x <- rnorm(100)
  mu <- rep(0,length(x))
  sd <- rep(1,length(x))
  sumDnorm(x,mu,sd)
}
\seealso{
\code{sum(dnorm(x, mean, sd, log=TRUE))}
}
