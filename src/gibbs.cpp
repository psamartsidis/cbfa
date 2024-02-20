
#include "RcppArmadillo.h"
#include "pgcpp.h"
#include "gigcpp.h"
#include "utils.h"
#include "rng.h"
#include <random>
#include <vector>
#include <iostream>
#ifdef _OPENMP
  #include <omp.h>
#endif
using namespace Rcpp;
using namespace arma;



double factor_variance( const arma::vec &, double , double );
int multinomial_lse (const arma::vec &);


/* Polya-gamma latent variables */
void pg_gibbs( const arma::ucube &n, const arma::cube &Lf_binom, arma::cube &omega, const arma::uvec &Timings )
{
  
  int nUnits = omega.n_cols;
  int D2     = omega.n_slices;
  int d, t, i;
  
  for ( d=0 ; d<D2 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<Timings(i)-1 ; t++) {
        
        if (n(t,i,d)>0) {
          omega(t,i,d) = pgcpp( n(t,i,d), Lf_binom(t,i,d) );
        }
        
      }
    }
  }
  
}



/* CRT latent variables */ 
void crt_gibbs( const arma::vec &xi, const arma::uvec &Timings, const arma::ucube &w, const arma::ucube &z, 
                const arma::cube &Lf, arma::ucube &crt )
{
  
  int D      = w.n_slices;
  int nUnits = w.n_cols;
  double tmp0, tmp2;
  int tmp1;
  
  for ( int d=0 ; d<D ; d++ ){
    for ( int i=0 ; i<nUnits ; i++ ){
      for ( int t=0 ; t<Timings(i)-1 ; t++ ) {
        
        if (w(t,i,d)>0) {
          tmp1 = 0;
          for ( int p=0 ; p<z(t,i,d); p++ ) {
            tmp0 = w(t,i,d) * exp(Lf(t,i,d)) / xi(d);
            tmp2 = tmp0/(tmp0+p);
            tmp1 += R::rbinom(1,tmp2);
          }
          crt(t,i,d) = tmp1;
        } 
        
      }
    }
  }
  
}



/* Binomial factors */
void f_binom_gibbs( const arma::mat &L, arma::cube &f, const arma::cube &kappa, const arma::mat &psi, 
                    const arma::cube &omega, const arma::uvec &Timings, const arma::uvec &nControls)
{
  int nTimes = f.n_rows;
  int nUnits = L.n_rows;
  int P      = f.n_cols;
  int D      = f.n_slices;
  int p, d, i, t, idx, nUnitsTime;
  arma::vec f_new(P);
  arma::vec f_mu(P);
  arma::mat f_prec(P,P);
  
  /* Repeat for all outcomes and time points */ 
  for ( d=0 ; d<D ; d++ ){
    for ( t=0 ; t<nTimes ; t++ ){
      
      /* Sub-matrices */
      nUnitsTime = nControls(t);
      arma::mat L_tmp( P, nUnitsTime ); 
      arma::vec kappa_tmp( nUnitsTime ); 
      arma::mat omega_tmp( nUnitsTime, nUnitsTime ); omega_tmp.fill(0.0);
      idx = 0;
      for (i=0 ; i<nUnits ; i++) {
        if ( (t+1)<Timings(i) ) {
          kappa_tmp(idx) = kappa(t,i,d);
          omega_tmp(idx,idx) = omega(t,i,d);
          for (p=0 ; p<P ; p++) {
            L_tmp(p,idx) = L(i,p);
          }
          idx += 1;
        }
      }
      
      /* Matrices for Rue's method */
      f_prec = L_tmp * omega_tmp * (L_tmp.t()); 
      /* Matrix omega_tmp above can be avoided. To improve. */
      for (p=0 ; p<P ; p++) {
        f_prec(p,p) += 1.0/psi(p,d);
      }
      f_mu   = L_tmp * kappa_tmp;
      
      /* Draw the new value */
      f_new = rueMVnorm( f_mu, f_prec, f.slice(d).row(t).t() );
      /* Lines below are redundant. We introduce them to add 'try catch' statements above or check for NAs/Infs  */
      for (p=0 ; p<P ; p++) {
        f(t,p,d) = f_new(p);
      } 
      
    }
  }
}



/* Normal factors */
void f_normal_gibbs( const arma::uvec &nControls, const arma::uvec &Timings, const arma::cube &y, const arma::mat &L, 
                     const arma::mat &sigma, const arma::mat &psi, arma::cube &f )
{
  int P      = f.n_cols;
  int nTimes = f.n_rows;
  int D      = f.n_slices;
  int nUnits = L.n_rows;
  int i, d, t, p, idx, nUnitsTime;
  arma::vec f_new(P);
  arma::vec f_mu(P);
  arma::mat f_prec(P,P);
  
  
  /* Repeat for all time points and outcomes */
  for ( d=0 ; d<D ; d++ ){
    for ( t=0 ; t<nTimes ; t++ ) {
      
      /* Likelihood contribution */
      nUnitsTime = nControls(t);
      arma::mat L_tmp( P, nUnitsTime );
      arma::vec y_tmp( nUnitsTime );
      arma::mat sigma_tmp( nUnitsTime, nUnitsTime ); sigma_tmp.fill(0.0);
      idx = 0;
      for (i=0 ; i<nUnits ; i++) {
        if ( (t+1)<Timings(i) ) {
          y_tmp(idx)         = y(t,i,d)/sigma(i,d);
          sigma_tmp(idx,idx) = 1.0/sigma(i,d);
          for (p=0 ; p<P ; p++) {
            L_tmp(p,idx) = L(i,p);
          }
          idx += 1;
        }
      }
      f_prec = L_tmp * sigma_tmp * (L_tmp.t());
      f_mu   = L_tmp * y_tmp;
      
      /* Prior contribution */
      for (p=0 ; p<P ; p++) {
        f_prec(p,p) += 1.0/psi(p,d);
      }
      
      /* Draw the new value */
      f_new = rueMVnorm( f_mu, f_prec, f.slice(d).row(t).t() );
      for (p=0 ; p<P ; p++) {
          f(t,p,d) = f_new(p);
      }
      
    } /* End of loop for time points */
  }   /* End of loop for outcomes */
  
}



/* Normal variance parameters */
void sigma_gibbs( const arma::cube &y, const arma::cube &Lf, const arma::uvec &Timings, arma::mat &sigma, double SMAX )
{
  int nUnits = y.n_cols;
  int D      = y.n_slices;
  int i, t, d, nTimesUnit;
  double alpha, beta, tmp;
  
  /* Repeat for all outcomes and units */
  for ( d=0 ; d<D ; d++ ){
    for ( i=0 ; i<nUnits ; i++ ){
      
      nTimesUnit = Timings(i) - 1;
      alpha      = 0.5*nTimesUnit - 1.0;
      beta       = 0.0;
      for (t=0 ; t<nTimesUnit ; t++) {
        beta += ( y(t,i,d) - Lf(t,i,d) ) * ( y(t,i,d) - Lf(t,i,d) );
      }
      beta *= 0.5;
      tmp   = 1.0/R::rgamma(alpha,1.0/beta);
      if (tmp<=SMAX) {
        sigma(i,d) = tmp;
      }
      
    }
  }
  
}



/* Variance of factor parameters */ 
void psi_gibbs( const arma::cube &f_normal, const arma::cube &f_binom, const arma::cube &f_negbin, 
                arma::mat &psi_normal, arma::mat &psi_binom, arma::mat &psi_negbin )
{
  int P      = f_normal.n_cols;
  int nTimes = f_normal.n_rows;
  int D1     = f_normal.n_slices;
  int D2     = f_binom.n_slices;
  int D3     = f_negbin.n_slices;
  int d, p;
  int idx=-1;
  double shape = 0.5*nTimes - 1.0;
  
  /* Normal factors */
  for ( d=0 ; d<D1 ; d++ ){
    idx += 1;
    if ( idx>0 ){
      for ( p=0 ; p<P ; p++ ){
        psi_normal(p,d) = factor_variance( f_normal.slice(d).col(p), shape , psi_normal(p,d) );

      }
    }
  }
  
  /* Binomial factors */
  for ( d=0 ; d<D2 ; d++ ){
    idx += 1;
    if ( idx>0 ){
      for ( p=0 ; p<P ; p++ ){
        psi_binom(p,d) = factor_variance( f_binom.slice(d).col(p), shape , psi_binom(p,d) );
        
      }
    }
  }
  
  /* NB factors */
  for ( d=0 ; d<D3 ; d++ ){
    idx += 1;
    if ( idx>0 ){
      for ( p=0 ; p<P ; p++ ){
        psi_negbin(p,d) = factor_variance( f_negbin.slice(d).col(p), shape , psi_negbin(p,d) );
        
      }
    }
  }
  
}



/* Truncated factor variance draw */
double factor_variance( const arma::vec &f, double shape, double psi0 )
{
  
  int nTimes = f.n_elem;
  int t;
  double psi_max = 1.0;
  double rate = 0.0;
  for ( t=0 ; t<nTimes ; t++ ) {
    rate += f(t)*f(t);
  }
  rate *= 0.5;
  double tmp = 1.0/R::rgamma(shape,1.0/rate);
  double psi;
  if ( tmp < psi_max ) {
    psi = tmp;
  } else {
    psi = psi0;
  }
  return(psi);
    
}



/* Loadings, conjugate model */ 
void L_gibbs( arma::mat &L, const arma::mat &theta ,const arma::cube &f_normal, const arma::cube &f_binom, const arma::cube &y, 
              const arma::mat &sigma, const arma::cube &omega, const arma::cube &kappa, const arma::uvec &Timings )
{
  int nUnits = L.n_rows;
  int P      = L.n_cols;
  int D1     = f_normal.n_slices;
  int D2     = f_binom.n_slices;
  int i, t, p, d, nTimesUnit;
  arma::vec Lmu(P), Lnew(P);
  arma::mat Lprec(P,P);
  
  /* Repeat for all units */
  for ( i=0 ; i<nUnits ; i++ ){
    
    Lmu.fill(0.0);
    Lprec.fill(0.0);
    nTimesUnit = Timings(i)-1;
    arma::mat f_tmp1( P, nTimesUnit );
    arma::mat f_tmp2( P, nTimesUnit );
    arma::vec y_tmp( nTimesUnit );

    /* Normal likelihood contribution */
    for ( d=0 ; d<D1 ; d++ ) {
      f_tmp1.fill(0.0);
      f_tmp2.fill(0.0);
      y_tmp.fill(0.0); 
      for ( t=0 ; t<nTimesUnit ; t++ ) {
        y_tmp(t) = y(t,i,d);
        for ( p=0 ; p<P ; p++ ) {
          f_tmp1(p,t) = f_normal(t,p,d)/sigma(i,d);
          f_tmp2(p,t) = f_normal(t,p,d);
        }
      }
      Lprec += f_tmp2 * (f_tmp1.t());
      Lmu   += f_tmp1 * y_tmp;
    }
    
    /* Binomial likelihood contribution */ 
    for ( d=0 ; d<D2; d++ ) {
      f_tmp1.fill(0.0);
      f_tmp2.fill(0.0);
      y_tmp.fill(0.0) ;
      for ( t=0 ; t<nTimesUnit ; t++ ){
        y_tmp(t)     = kappa(t,i,d);
        for ( p=0 ; p<P ; p++ ){
          f_tmp1(p,t) = f_binom(t,p,d)*omega(t,i,d);
          f_tmp2(p,t) = f_binom(t,p,d);
        }
      }
      Lprec += f_tmp2 * (f_tmp1.t());
      Lmu   += f_tmp2 * y_tmp;
    }
    
    /* Prior contribution */ 
    for ( p=0 ; p<P ; p++ ) {
      Lprec(p,p) += 1.0/theta(i,p);
    }
    
    /* Draw the new value */
    Lnew = rueMVnorm( Lmu, Lprec, L.row(i).t() );
    for (p=0 ; p<P ; p++) {
      L(i,p) = Lnew(p);
    }
    
  } /* End of loop for units */
  
}
    
 
  
/* TPB shrinkage parameters */
arma::vec tpb_gibbs( const arma::mat &L, arma::mat &theta, arma::mat &delta, arma::vec &phi, arma::vec &tau, double eta, 
                     double gamma, const arma::vec &tpb_prior )
{
  int nUnits = L.n_rows;
  int P      = L.n_cols;
  double tmp, tmp1, tmp2;
  int i, p;
  arma::vec output(2); output(0) = eta; output(1) = gamma;
  
  
  /* Local shrinkage parameters */
  for (i=0 ; i<nUnits ; i++) {
    for (p=0 ; p<P ; p++) {
      /* theta */
      tmp1       = 2.0*delta(i,p);
      tmp2       = L(i,p)*L(i,p);
      if ( (tmp1!=R_PosInf) && (tmp2>0) ) {
        tmp = gigcpp( tpb_prior(0)-0.5, tmp2, tmp1 );
        if ( (tmp>0) && (tmp!=R_PosInf) ) {
          theta(i,p) = tmp; 
        }
      }
      /* delta */
      tmp1       = tpb_prior(0) + tpb_prior(1);
      tmp2       = theta(i,p) + phi(p);
      tmp        = R::rgamma(tmp1,1.0/tmp2);
      if ( (tmp>0) && (tmp!=R_PosInf) ) {
        delta(i,p) = tmp; 
      }
    }
  }
  
  /* Column shrinkage parameters */
  for (p=0 ; p<P ; p++) {
    /* phi */
    tmp1 = nUnits*tpb_prior(1) + tpb_prior(2);
    tmp2 = tau(p);
    for (i=0 ; i<nUnits ; i++) {
      tmp2 += delta(i,p);
    }
    tmp =  R::rgamma(tmp1,1.0/tmp2);
    if ( (tmp>0) && (tmp!=R_PosInf) ) {
      phi(p) = tmp; 
    }
    /* tau */
    tmp1   = tpb_prior(2) + tpb_prior(3);
    tmp2   = phi(p) + eta;
    tmp    = R::rgamma(tmp1,1.0/tmp2);
    if ( (tmp>0) && (tmp!=R_PosInf) ) {
      tau(p) = tmp; 
    }
  }
  
  /* Global shrinkage parameters */
  /* eta */
  tmp1 = P*tpb_prior(3) + tpb_prior(4);
  tmp2 = gamma;
  for (p=0 ; p<P ; p++) {
    tmp2 += tau(p);
  }
  tmp = R::rgamma(tmp1,1.0/tmp2);
  if ( (tmp>0) && (tmp!=R_PosInf) ) {
    output(0) = tmp; 
  }
  /* gamma */
  tmp1      = tpb_prior(4) + tpb_prior(5);
  tmp2      = output(0) + tpb_prior(6);
  tmp       = R::rgamma(tmp1,1.0/tmp2);
  if ( (tmp>0) && (tmp!=R_PosInf) ) {
    output(1) = tmp; 
  }
  
  
  return(output);
}



/* Regression coefficients, conjugate model */
void beta_normal_gibbs( const arma::cube &X, const arma::cube &y, arma::mat &beta, const arma::uvec &Timings, 
                        const arma::mat &sigma )
{
  int J      = X.n_slices; 
  int nUnits = X.n_cols;
  int D1     = y.n_slices;
  int d, i, nTimesUnit, t, j;
  arma::vec betaMu(J), betaNew(J);
  arma::mat betaPrec(J,J);
  
  
  /* Repeat for all the outcomes */
  for ( d=0 ; d<D1 ; d++ ) {
    
    betaMu.fill(0.0);
    betaPrec.fill(0.0);
    
    /* Loops over units */ 
    for ( i=0 ; i<nUnits ; i++ ) {
      /* Unit data */ 
      nTimesUnit = Timings(i) - 1;
      arma::mat X_tmp1(nTimesUnit,J);
      arma::mat X_tmp2(nTimesUnit,J);
      arma::vec y_tmp(nTimesUnit); 
      for ( t=0 ; t<nTimesUnit ; t++ ) {
        y_tmp(t) = y(t,i,d);
        for ( j=0 ; j<J ; j++ ) {
          X_tmp1(t,j) = X(t,i,j);
        }
      }
      X_tmp2 = X_tmp1/sigma(i,d);
      /* Add to the mean/precision */ 
      betaMu   += X_tmp2.t() * y_tmp;
      betaPrec += X_tmp1.t() * X_tmp2;
    }
    
    /* Prior contribution */ 
    for ( j=0 ; j<J ; j++ ) {
      betaPrec(j,j) += 0.001;
    }
    
    /* Draw new value */
    betaNew = rueMVnorm( betaMu, betaPrec, beta.col(d) );
    for ( j=0 ; j<J ; j++ ) {
      beta(j,d) = betaNew(j);
    }
    
  } /* End of loop over outcomes */
  
}



/* Regression coefficients, conjugate model */
void beta_binom_gibbs( const arma::cube &X, const arma::cube &kappa, const arma::cube &omega, arma::mat &beta, 
                          const arma::uvec &Timings )
{
  int J      = X.n_slices; 
  int nUnits = X.n_cols;
  int D2     = omega.n_slices;
  int d, i, nTimesUnit, t, j;
  arma::vec betaMu(J), betaNew(J);
  arma::mat betaPrec(J,J);
  
  
  /* Repeat for all the outcomes */
  for ( d=0 ; d<D2 ; d++ ) {
    
    betaMu.fill(0.0);
    betaPrec.fill(0.0);
    
    /* Loops over units */ 
    for ( i=0 ; i<nUnits ; i++ ) {
      /* Unit data */ 
      nTimesUnit = Timings(i) - 1;
      arma::mat X_tmp1(nTimesUnit,J);
      arma::mat X_tmp2(nTimesUnit,J);
      arma::vec kappa_tmp(nTimesUnit); 
      for ( t=0 ; t<nTimesUnit ; t++ ) {
        kappa_tmp(t) = kappa(t,i,d);
        for ( j=0 ; j<J ; j++ ) {
          X_tmp1(t,j) = X(t,i,j);
          X_tmp2(t,j) = X(t,i,j)*omega(t,i,d);
        }
      }
      /* Add to the mean/precision */ 
      betaMu   += X_tmp1.t() * kappa_tmp;
      betaPrec += X_tmp1.t() * X_tmp2;
    }
    
    /* Prior contribution */ 
    for ( j=0 ; j<J ; j++ ) {
      betaPrec(j,j) += 0.001;
    }
    
    /* Draw new value */
    betaNew = rueMVnorm( betaMu, betaPrec, beta.col(d) );
    for ( j=0 ; j<J ; j++ ) {
      beta(j,d) = betaNew(j);
    }
    
  } /* End of loop over outcomes */
    
}



/* Alternative sampler for the factor variance */ 
void psi_gibbs_latent( const arma::cube &f_normal, const arma::cube &f_binom, const arma::cube &f_negbin, 
                arma::mat &psi_normal, arma::mat &psi_binom, arma::mat &psi_negbin )
{
  int P      = f_normal.n_cols;
  int nTimes = f_normal.n_rows;
  int D1     = f_normal.n_slices;
  int D2     = f_binom.n_slices;
  int D3     = f_negbin.n_slices;
  int D      = D1+D2+D3;
  int d, p, t;
  int idx, MAX;
  double shape  = 0.5*nTimes - 1.0;
  double tmp1, tmp2, rate;
  arma::vec qvec(D);
  
  
  /* Find the factor with variance 1 */ 
  for ( p=0 ; p<P ; p++ ) {
    
    qvec.fill(0.0);
    idx = -1;
    
    /* Normal outcomes */
    for ( d=0 ; d<D1 ; d++ ) {
      idx  += 1;
      tmp1 = 0.0;
      for ( t=0 ; t<nTimes ; t++ ) { tmp1 += f_normal(t,p,d)*f_normal(t,p,d); }
      rate = 0.5*tmp1;
      tmp1 = R::pgamma( 1.0, shape, 1/rate, 0, 1 );
      tmp2 = R::dgamma( 1.0, shape, 1/rate, 1);
      for (t=0 ; t<D ; t++) { if (t==idx) { qvec(t) +=  tmp2; } else { qvec(t) += tmp1; } }
    }
    
    /* Binomial outcomes */ 
    for ( d=0 ; d<D2 ; d++ ) {
      idx  += 1;
      tmp1 = 0.0;
      for ( t=0 ; t<nTimes ; t++ ) { tmp1 += f_binom(t,p,d)*f_binom(t,p,d); }
      rate = 0.5*tmp1;
      tmp1 = R::pgamma( 1.0, shape, 1/rate, 0, 1 );
      tmp2 = R::dgamma( 1.0, shape, 1/rate, 1);
      for (t=0 ; t<D ; t++) { if (t==idx) { qvec(t) +=  tmp2; } else { qvec(t) += tmp1; } }
    }
    
    /* Count outcomes */
    for ( d=0 ; d<D3 ; d++ ) {
      idx  += 1;
      tmp1 = 0.0;
      for ( t=0 ; t<nTimes ; t++ ) { tmp1 += f_negbin(t,p,d)*f_negbin(t,p,d); }
      rate = 0.5*tmp1;
      tmp1 = R::pgamma( 1.0, shape, 1/rate, 0, 1 );
      tmp2 = R::dgamma( 1.0, shape, 1/rate, 1);
      for (t=0 ; t<D ; t++) { if (t==idx) { qvec(t) +=  tmp2; } else { qvec(t) += tmp1; } }
    }
    
    /* Draw M */
    MAX = multinomial_lse(qvec);
    
    /* Draw the variance of the factors according to M  */ 
    idx = -1; 
    
    /* Normal outcomes */
    for ( d=0; d<D1 ; d++ ){
      idx += 1;
      if (MAX==idx) { psi_normal(p,d) = 1.0; } else {
        psi_normal(p,d) = factor_variance( f_normal.slice(d).col(p), shape , psi_normal(p,d) );
      }
    }
    
    /* Binomial outcomes */
    for ( d=0; d<D2 ; d++ ){
      idx += 1;
      if (MAX==idx) { psi_binom(p,d) = 1.0; } else {
        psi_binom(p,d) = factor_variance( f_binom.slice(d).col(p), shape , psi_binom(p,d) );
      }
    }
    
    /* Normal outcomes */
    for ( d=0; d<D3 ; d++ ){
      idx += 1;
      if (MAX==idx) { psi_negbin(p,d) = 1.0; } else {
        psi_negbin(p,d) = factor_variance( f_negbin.slice(d).col(p), shape , psi_negbin(p,d) );
      }
    }
    
  } /* End of loop for factors */ 
  
  
}



/* Multinomial using the LOG-SUM-EXP trick */
int multinomial_lse (const arma::vec &qvec) {
  int K = qvec.n_elem;
  int k, idx;
  
  /* Find the maximum */
  double MX = qvec(0);
  for (k=1 ; k<K ; k++) {
    if (qvec(k)>MX) {
      MX = qvec(k);
    }
  }
  
  /* Find the log of the sum */
  double SUM = 0.0;
  double tmp;
  for (k=0 ; k<K ; k++) {
    tmp = exp( qvec(k)-MX );
    SUM += tmp;
  }
  SUM = log(SUM) + MX;
  
  /* Cumulative */
  arma::vec cumul(K);
  for (k=0 ; k<K ; k++) {
    cumul(k) = exp( qvec(k) - SUM );
  }
  for (k=1 ; k<K ; k++) {
    cumul(k) += cumul(k-1);
  }
  
  /* draw from the multinomial */
  idx = -1;
  double u = R::runif(0.0,1.0);
  int FLAG = 0;
  while( FLAG==0 ) {
    idx += 1;
    if (u<=cumul(idx)) {
      FLAG = 1;
    }
  }
  
  return(idx);
}



/* Polya-gamma latent variables in parallel */
void pg_gibbs_parallel( const arma::ucube &omega_omp_idx, std::vector<std::mt19937> &generator,
                        const arma::cube &Lf, arma::cube &omega, const arma::uvec &omega_omp_loop )
{
  int i;
  int nCores = omega_omp_idx.n_slices;
  
  #ifdef _OPENMP 
  #pragma omp parallel for 
  #endif
  for ( i=0 ; i<nCores ; i++ ) {
    
    unsigned int j;
    int idx_unit, idx_time, idx_outcome, idx_n;
    for ( j=0 ; j<omega_omp_loop(i) ; j++ ) {
      idx_outcome = omega_omp_idx(j,0,i); 
      idx_unit    = omega_omp_idx(j,1,i); 
      idx_time    = omega_omp_idx(j,2,i);
      idx_n       = omega_omp_idx(j,3,i);
      omega(idx_time,idx_unit,idx_outcome) = rng_polya_gamma( generator[i], idx_n, Lf(idx_time,idx_unit,idx_outcome) );
    }
    
  }
  
}



/* CRT latent variables in parallel */ 
void crt_gibbs_parallel( const arma::ucube &crt_omp_idx, std::vector<std::mt19937> &generator, const arma::vec &xi, 
                         const arma::cube &Lf, arma::ucube &crt, const arma::uvec &crt_omp_loop )
{
  
  int i;
  int nCores = crt_omp_idx.n_slices;

  
  #ifdef _OPENMP 
  #pragma omp parallel for 
  #endif
  for ( i=0 ; i<nCores ; i++ ) {
    
    double tmp0, tmp2;
    unsigned int p, j;
    int tmp1, idx_unit, idx_time, idx_outcome, idx_w, idx_z;
    for ( j=0 ; j<crt_omp_loop(i) ; j++ ) {
      idx_outcome = crt_omp_idx(j,0,i); 
      idx_unit    = crt_omp_idx(j,1,i); 
      idx_time    = crt_omp_idx(j,2,i);
      idx_w       = crt_omp_idx(j,3,i);
      idx_z       = crt_omp_idx(j,4,i);
      tmp0        = idx_w * exp( Lf(idx_time,idx_unit,idx_outcome) ) / xi(idx_outcome) ;
      tmp1        = 0;
      for ( p=0 ; p<idx_z ; p++ ){
        tmp2 = tmp0/(tmp0+p);
        tmp1 += rng_binomial( generator[i], 1, tmp2);
      }
      crt(idx_time,idx_unit,idx_outcome) = tmp1;
    }
    
  }
      
}
