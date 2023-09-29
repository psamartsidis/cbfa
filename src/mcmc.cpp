#include "RcppArmadillo.h"
#include <stdio.h>
#include <omp.h>


using namespace std;
using namespace Rcpp;
using namespace arma;



void mcmc( const arma::mat &y, int z ) {
  
  int nCores = omp_get_num_threads();
  
  int i;
  for ( i=0 ; i<z ; i++ ) {
    cout << "Iteration: " << i+1 << endl;
  }
  
  cout << "Number of cores: " << nCores << endl;
  
  
}