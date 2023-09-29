
#include "RcppArmadillo.h"
using namespace Rcpp;
using namespace arma;



/* Rue JRSSB 2001 method to draw from MVN */
arma::vec rueMVnorm(const arma::vec &b, const arma::mat &Q0, const arma::vec &original ) {
  int N = Q0.n_cols;
  arma::mat Q(N,N);
  arma::vec b_new = original;
  int i, j;
  
  
  /* Choleski of Q */
  bool flag = chol(Q,Q0);
  if ( flag ) {
    
    Q = Q.t();
    
    /* Find v: Lv=b */
    arma::vec v(N); v.fill(0.0);
    v(0) = b(0)/Q(0,0);
    for (i=1 ; i<(N) ; i++) {
      v(i) = b(i); 
      for (j=0 ; j<i ; j++) {
        v(i) -= Q(i,j)*v(j);
      }
      v(i) /= Q(i,i);
    }
    
    
    /* Find m: Lm = v */
    arma::vec m(N); m.fill(0.0);
    m(N-1) = v(N-1)/Q(N-1,N-1);
    for (i=(N-2) ; i>=0 ; --i) {
      m(i) = v(i);
      for (j=(i+1) ; j<N ; j++) {
        m(i) -= Q(j,i)*m(j);
      }
      m(i) /= Q(i,i);
    }
    
    
    /* Generate z from N(0,I) */
    arma::vec z(N); z.fill(0);
    for (i=0 ; i<N ; i++) { 
      z(i) = R::rnorm(0,1);
    }
    
    
    /* Find w: Lw=z */
    arma::vec w(N); w.fill(0.0);
    w(N-1) = z(N-1)/Q(N-1,N-1);
    for (i=(N-2) ; i>=0 ; --i) {
      w(i) = z(i);
      for (j=(i+1) ; j<N ; j++) {
        w(i) -= Q(j,i)*w(j);
      }
      w(i) /= Q(i,i);
    }
    
    
    /* Add m and w */
    for (i=0 ; i<N ; i++) {
      b_new(i) = m(i) + w(i);
    }
    
  } else {
    
    /* Message */
    Rcout << "Decomposition failed: " << 1 << "\n";
    R_FlushConsole();
    
  }
  
  /* Output */
  return(b_new);
  
}



arma::umat offset_save( const arma::ucube &n, const arma::uvec &Timings )
{
  int nUnits = n.n_cols;
  int nTimes = n.n_rows;
  int D      = n.n_slices;
  int d, t, i;
  
  /* Find out how many PG variables are drawn in total */
  int offset_n = 0;
  if ( D!=0 ) {
    for ( d=0 ; d<D ; d++ ) {
      for ( i=0 ; i<nUnits ; i++ ){
        for ( t=0 ; t<Timings(i)-1 ; t++ ) {
          if ( n(t,i,d)>0 ) {
            offset_n += 1; 
          }
        }
      }
    }
  }
  
  /* Find which ones they are */
  arma::umat offset_positive(offset_n,3);
  int idx = 0;
  for ( d=0 ; d<D ; d++ ) {
    for ( i=0 ; i<nUnits ; i++ ){
      for ( t=0 ; t<Timings(i)-1 ; t++ ) {
        if ( n(t,i,d)>0 ) {
          offset_positive(idx,0) = t;
          offset_positive(idx,1) = i;
          offset_positive(idx,2) = d;
          idx += 1;
        }
      }
    }
  }
 
  /*  Randomly sample at most 100 to save */
  int save_n = 100;
  if (offset_n<save_n) {
    save_n = offset_n;
  }
  
  
  /* Store their indices */
  arma::umat offset_idx(save_n,3);
  if (save_n>0) {
    arma::uvec FLAG = randperm(offset_n,save_n);
    for (i=0 ; i<save_n ; i++) {
      offset_idx(i,0) = offset_positive(FLAG(i),0);
      offset_idx(i,1) = offset_positive(FLAG(i),1);
      offset_idx(i,2) = offset_positive(FLAG(i),2);
    }
  }
  

  return(offset_idx);
}



void save_state( int mcmc_idx, const arma::umat &omega_idx, const arma::cube &omega, arma::mat &omega_mcmc,
                 const arma::vec &xi, arma::mat &xi_mcmc, const arma::umat &crt_idx, const arma::ucube &crt, arma::umat &crt_mcmc,
                 const arma::cube &f_binom, arma::cube & f_binom_mcmc,  const arma::cube &f_negbin, arma::cube &f_negbin_mcmc, 
                 const arma::cube &f_normal, arma::cube &f_normal_mcmc, const arma::mat &sigma, arma::cube &sigma_mcmc,
                 const arma::mat &psi_normal, const arma::mat &psi_binom, const arma::mat &psi_negbin, arma::cube &psi_mcmc,
                 const arma::mat &L, arma::cube &L_mcmc, const arma::vec &tau, arma::mat &tau_mcmc, const arma::vec &phi, 
                 arma::mat &phi_mcmc, double eta, arma::vec &eta_mcmc, double gamma, arma::vec &gamma_mcmc )
{
  int i, t, p, d, idx;
  int omega_n = omega_idx.n_rows;
  int crt_n   = crt_idx.n_rows;
  int P       = f_binom.n_cols;
  int nTimes  = f_binom.n_rows;
  int D1      = f_normal.n_slices;
  int D2      = f_binom.n_slices;
  int D3      = f_negbin.n_slices;
  

  /* PG latent variables */
  for ( i=0 ; i<omega_n ; i++ ) {
    omega_mcmc(i,mcmc_idx) = omega( omega_idx(i,0), omega_idx(i,1), omega_idx(i,2) );
  } 
  
  /* NB dispersion parameters */
  xi_mcmc.col(mcmc_idx) = xi;
  
  /* CRT latent variables */
  for ( i=0 ; i<crt_n ; i++ ) {
    crt_mcmc(i,mcmc_idx) = crt( crt_idx(i,0), crt_idx(i,1), crt_idx(i,2) );
  } 
  
  /* Binomial factors */
  for ( d=0 ; d<D2 ; d++ ){
    idx = d*P;
    for ( p=0 ; p<P ; p++ ) {
      for ( t=0 ; t<nTimes ; t++ ) {
        f_binom_mcmc(t,idx+p,mcmc_idx) = f_binom(t,p,d);
      }
    }
  }
  
  /* NB factors */
  for ( d=0 ; d<D3 ; d++ ){
    idx = d*P;
    for ( p=0 ; p<P ; p++ ) {
      for ( t=0 ; t<nTimes ; t++ ) {
        f_negbin_mcmc(t,idx+p,mcmc_idx) = f_negbin(t,p,d);
      }
    }
  }
  
  /* Normal factors */
  for ( d=0 ; d<D1 ; d++ ){
    idx = d*P;
    for ( p=0 ; p<P ; p++ ) {
      for ( t=0 ; t<nTimes ; t++ ) {
        f_normal_mcmc(t,idx+p,mcmc_idx) = f_normal(t,p,d);
      }
    }
  }
  
  /* Normal variance parameters */
  sigma_mcmc.slice(mcmc_idx) = sigma;
  
  /* Factor variance parameters */
  idx = 0;
  for ( d=0 ; d<D1 ; d++ ){
    for ( p=0 ; p<P ; p++ ) {
      psi_mcmc(p,idx,mcmc_idx) = psi_normal(p,d);
    }
    idx += 1;
  }
  for ( d=0 ; d<D2 ; d++ ){
    for ( p=0 ; p<P ; p++ ) {
      psi_mcmc(p,idx,mcmc_idx) = psi_binom(p,d);
    }
    idx += 1;
  }
  for ( d=0 ; d<D3 ; d++ ){
    for ( p=0 ; p<P ; p++ ) {
      psi_mcmc(p,idx,mcmc_idx) = psi_negbin(p,d);
    }
    idx += 1;
  }
  
  /* Loadings */
  L_mcmc.slice(mcmc_idx) = L;
  
  /* Loadings shrinkage parameters */
  tau_mcmc.col(mcmc_idx) = tau;
  phi_mcmc.col(mcmc_idx) = phi;
  eta_mcmc(mcmc_idx)     = eta;
  gamma_mcmc(mcmc_idx)   = gamma;

}



void update_Lf( const arma::mat &L, const arma::cube &f_normal, const arma::cube &f_binom, 
                const arma::cube &f_negbin, arma::cube &Lf_normal, arma::cube &Lf_binom, 
                arma::cube &Lf_negbin )
{
  int D1 = f_normal.n_slices;
  int D2 = f_binom.n_slices;
  int D3 = f_negbin.n_slices;
  int d;
  
  /* Normal */
  for (d=0 ; d<D1 ; d++) {
    Lf_normal.slice(d) = f_normal.slice(d) * L.t();
  }
  /* Binomial */
  for (d=0 ; d<D2 ; d++) {
    Lf_binom.slice(d) = f_binom.slice(d) * L.t();
  }
  /* Count */
  for (d=0 ; d<D3 ; d++) {
    Lf_negbin.slice(d) = f_negbin.slice(d) * L.t();
  }
  
}



void update_stepsize (int mcmc_idx, int nBurn, double target_lower, double target_upper, 
                      const arma::umat &accept, arma::vec &stepsize, arma::vec &average )
{
  int window = accept.n_rows - 1;
  int nVars  = accept.n_cols;
  int i,j;
  double acceptance_ratio;
  
  for ( j=0 ; j<nVars ; j++ ){
    
    /* Find the AR over the last window iterations */
    acceptance_ratio = 0.0;
    for ( i=0 ; i<window ; i++ ){
      acceptance_ratio += accept(i,j);
    }
    acceptance_ratio = acceptance_ratio/window;
    
    /* Adjust depending on the acceptance ratio */
    if (acceptance_ratio>target_upper) {
      stepsize(j) *= 1.01;
    }
    if (acceptance_ratio<target_lower) {
      stepsize(j) *= 0.99;
    }
    
    /* Add to the running summation */
    if ( mcmc_idx > (0.5*nBurn) ) {
      average(j) += 2.0*stepsize(j)/nBurn;
    }
    
  }
  
  /* If it is the last iteration of burnin, set to the average */
  if ( mcmc_idx==nBurn ) {
    stepsize = average;
  }
  
}



void acceptance_ratios( int nMCMC, int nBurn, const arma::umat &xi_accept, arma::vec &xi_ar, const arma::umat &f_negbin_accept, 
                        arma::mat &f_negbin_ar, const arma::umat &L_accept,  arma::vec &L_ar ) 
{
  double nDraws = (double) nMCMC - nBurn;
  int i, idx, d;
  int window = xi_accept.n_rows-1;
  int nTimes = f_negbin_ar.n_cols;
  int nUnits = L_accept.n_cols;
  int D3 = xi_accept.n_cols;
  /* NB dispersion */
  for ( i=0 ; i<D3 ; i++ ) {
    xi_ar(i) = xi_accept(window,i) / nDraws;
  }
  /* NB factors */
  for ( d=0 ; d<D3 ; d++ ) {
    for ( i=0 ; i<nTimes ; i++ ) {
      idx = d*nTimes + i;
      f_negbin_ar(d,i) = f_negbin_accept(window,idx) / nDraws;
    }
  }
  /* Loadings */ 
  for ( i=0 ; i<nUnits ; i++ ){
    L_ar(i) = L_accept(window,i) / nDraws;
  }
  

}