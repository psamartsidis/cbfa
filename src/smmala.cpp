
#include "RcppArmadillo.h"
using namespace arma;


double grad_xi( const arma::mat &,  double xi, const arma::uvec &, const arma::umat &, const arma::umat & );
void grad_factors(const arma::uvec &, const arma::umat &, const arma::mat &, const arma::mat &, const arma::mat &, 
                  const arma::umat &, arma::mat &, arma::mat &, const arma::vec &, arma::mat &, const arma::vec & );
void hessian_factors( arma::cube &, arma::cube &, arma::cube &, const arma::mat &, const arma::uvec &, const arma::umat &, 
                      const arma::mat & , const arma::vec &, arma::vec &, const arma::vec &, arma::uvec &, bool );
void grad_loadings ( arma::mat &, const arma::uvec &, const arma::mat &, const arma::mat &,  const arma::cube &, 
                     const arma::mat &, const arma::cube &, const arma::cube &, const arma::cube &, const arma::ucube &, 
                     const arma::cube &, const arma::cube &, arma::cube &, const arma::cube &, const arma::ucube &, 
                     const arma::ucube &, const arma::ucube &, const arma::mat &, arma::cube &, 
                     const arma::cube &, const arma::cube &, arma::cube & );
void hessian_loadings( arma::vec &, arma::cube &, arma::cube &, arma::cube &, const arma::uvec &, const arma::mat &, 
                  const arma::cube &, const arma::mat &, const arma::cube &, const arma::ucube &, const arma::cube &,
                  const arma::cube &, const arma::ucube &, const arma::cube &, const arma::mat &, arma::uvec &, bool );


/* Barker update of the NB over-dispersion parameter */ 
/* We are working with the transformation y=log(xi) */
// [[Rcpp::export]]
void barker_dispersion( const arma::uvec &Timings, const arma::ucube &z, const arma::ucube &w, 
                        arma::vec &xi, const arma::vec &step, const arma::cube &Lf, arma::umat &xi_accept,
                        int mcmc_idx, int nBurn )
{
  int nUnits = Lf.n_cols;
  int nTimes = Lf.n_rows;
  int D      = Lf.n_slices;
  int window = xi_accept.n_rows-1;
  
  /* Some variables that will be used */
  int i, t;
  double xi_new, Z, b, grad_new, grad_old, barker_q, MH, u ,ratio, beta1, beta2, max1, max2; 
  double r_old, r_new, prob_old, prob_new;
  arma::mat mu(nTimes,nUnits);
  mu.fill(0.0);
  
  /* Repeat for all NB outcomes */ 
  for ( int d=0 ; d<D ; d++ ) {
    
    
    
    /* Mean function */ 
    for ( i=0 ; i<nUnits ; i++ ){
      for ( t=0 ; t<Timings(i)-1 ; t++ ) {
        mu(t,i) = exp(Lf(t,i,d));
      }
    }
    
    /* Evaluate gradient at current point */
    grad_old = grad_xi( mu, xi(d), Timings, w.slice(d), z.slice(d) );
    
    /* Propose new value */
    Z = step(d) * R::rnorm(0.0,1.0);
    ratio = 1.0/( 1.0 +exp(-Z*grad_old) );
    u     = R::runif(0.0,1.0);
    if (u<=ratio) {
      b = 1.0;
    } else {
      b = -1.0;
    }
    xi_new = exp( log(xi(d)) + b*Z ) ;
    
    
    /* Evaluate MH ratio if inside range */
    MH = 0.0;
    if ( (xi_new>0.0) && (xi_new<10.0) ) {
      
      /* Evaluate the gradient at the proposed point */
      grad_new = grad_xi( mu, xi_new, Timings, w.slice(d), z.slice(d) );
      
      /* Barker q-ratio */
      beta1 = -grad_new * ( log(xi(d)) - log(xi_new) );
      beta2 = -grad_old * ( log(xi_new) - log(xi(d)) );
      if (beta1>0.0) {
        max1 = beta1;
      } else {
        max1 = 0.0;
      }
      if (beta2>0.0) {
        max2 = beta2;
      } else {
        max2 = 0.0;
      }
      barker_q = -( max1 + log1p(exp(-fabs(beta1)))) + (max2 + log1p(exp(-fabs(beta2)))) ;
      
      /* Ratio of log-posteriors */ 
      prob_old = 1.0/(1.0+xi(d));
      prob_new = 1.0/(1.0+xi_new);
      for ( i=0 ; i<nUnits ; i++ ) {
        for ( t=0 ; t<Timings(i)-1 ; t++ ) {
          if ( w(t,i,d)>0 ) {
            r_old = w(t,i,d) * mu(t,i) / xi(d);
            r_new = w(t,i,d) * mu(t,i) / xi_new;
            MH += R::dnbinom(z(t,i,d), r_new, prob_new, true) - R::dnbinom(z(t,i,d), r_old, prob_old, true);
          }
        }
      }
      
      /* Prior contribution,  */
      MH += xi_new - xi(d);
      MH += barker_q;
      MH  = exp(MH);
      
    }
    
    /* Accept/reject */
    u = R::runif(0.0,1.0);
    if (u<=MH) {
      xi(d) = xi_new;
      if (mcmc_idx>nBurn) {
        xi_accept(window,d) += 1;
      } else {
        xi_accept(mcmc_idx%window,d) = 1;
      }
    } else {
      if (mcmc_idx<=nBurn) {
        xi_accept(mcmc_idx%window,d) = 0;
      }
    }
    
    
    
  }
}



/* Gradient for NB over-dispersion parameter */
/* We are working with the transformation y=log(xi) */
double grad_xi( const arma::mat &mu,  double xi, const arma::uvec &Timings, const arma::umat &w, const arma::umat &z )
{
  int nUnits = mu.n_cols;
  int nTimes = mu.n_rows;
  double grad=0.0;
  
  /* Constants */
  int i, t;
  arma::cube c_panel(nTimes,nUnits,2);
  c_panel.fill(0.0);
  arma::vec c_xi(4);
  c_xi(0) = log( xi + 1.0 );
  c_xi(1) = 1.0/xi;
  c_xi(2) = 1.0/(xi+1.0);
  c_xi(3) = c_xi(2) - c_xi(0)*c_xi(1);
  for (i=0 ; i<nUnits ; i++) {
    for (t=0 ; t<(Timings(i)-1) ; t++) {
      if (w(t,i)>0) {
        c_panel(t,i,1) = w(t,i) * mu(t,i);
        c_panel(t,i,0) = R::digamma( c_panel(t,i,1)/xi + z(t,i) ) - R::digamma( c_panel(t,i,1)/xi );
      }
    }
  }
  
  /* Likelihood contributions */
  double tmp1=0.0;
  double tmp2=0.0;
  double tmp3=0.0;
  for (i=0 ; i<nUnits ; i++) {
    for (t=0 ; t<(Timings(i)-1) ; t++) {
      if (w(t,i)>0) {
        tmp1 += c_panel(t,i,1)*c_panel(t,i,0);
        tmp2 += z(t,i);
        tmp3 += c_panel(t,i,1);
      }
    }
  }
  grad += -tmp1*c_xi(1) + tmp2*c_xi(2) - tmp3*c_xi(3);
  
  /* Prior contribution */
  /* This comes from the transformation y=log(xi) */
  grad += 1.0;
  
  return(grad);
}



/* SM-MALA update of Negative Binomial factors */
// [[Rcpp::export]]
void smmala_negbin_factors(const arma::mat &L, arma::cube &f, const arma::uvec &Timings, const arma::vec &xi, 
                           const arma::ucube &w, const arma::ucube &crt, const arma::mat &psi, const arma::cube &Lf, 
                           const arma::vec &step, arma::umat &f_accept, int mcmc_idx, int nBurn ) 
{
  int nUnits = L.n_rows;
  int nTimes = f.n_rows;
  int P      = f.n_cols;
  int D      = f.n_slices;
  int window = f_accept.n_rows-1;
  int d, i, p, t, idx;
  arma::mat  grad_old(nTimes,P),    grad_new(nTimes,P);
  arma::vec  det_new(nTimes),       det_old(nTimes);
  arma::cube G_new(P,P,nTimes),     G_old(P,P,nTimes);
  arma::cube Ginv_new(P,P,nTimes),  Ginv_old(P,P,nTimes);
  arma::cube Gchol_new(P,P,nTimes), Gchol_old(P,P,nTimes);
  /* Defining cube structures above spares us several calculations, with minimal RAM cost */
  arma::mat Lf_new(nTimes,nUnits), mu_new(nTimes,nUnits), c_new(nTimes,nUnits), f_new(nTimes,P);
  arma::mat Lf_old(nTimes,nUnits), mu_old(nTimes,nUnits), c_old(nTimes,nUnits), f_old(nTimes,P); 
  arma::mat mean_old(P,nTimes), mean_new(P,nTimes), Z(P,nTimes);
  arma::vec tmp(P), step_sq(nTimes), q(nTimes), MH(nTimes); 
  arma::mat scalar(1,1);
  double u;
  arma::uvec flag(nTimes); 
  
  
  /* Some dispersion-related terms used multiple times */
  arma::mat c_xi(2,D);
  for ( d=0 ; d<D ; d++ ){
    c_xi(0,d) = 1.0/xi(d);
    c_xi(1,d) = log( 1.0/(1.0+xi(d)) );
  }
 
 
  /* Repeat for all outcomes */
  for ( d=0 ; d<D ; d++ ) {
    
    f_old  = f.slice(d);
    Lf_old = Lf.slice(d);
    flag.fill(1);
    
    /* Gradient and Hessian at initial value */
    grad_factors( Timings, w.slice(d), f_old, L, Lf_old,  crt.slice(d), grad_old, mu_old, c_xi.col(d), c_old, psi.col(d) );
    hessian_factors( G_old, Gchol_old, Ginv_old, mu_old, Timings, w.slice(d), L , c_xi.col(d), det_old, psi.col(d), flag, true );
    
    /* Propose new values */
    for ( t=0 ; t<nTimes ; t++ ) {
      
      /* Stepsize */
      idx        = d*nTimes+t;
      step_sq(t) = 0.5*step(idx)*step(idx);
      
      /* The noise of the proposal */
      for (p=0 ; p<P ; p++) {
        tmp(p) = R::rnorm(0.0,1.0);
      }
      tmp = (Gchol_old.slice(t)) * tmp;
      for (p=0 ; p<P ; p++) {
        tmp(p) *= step(idx);
      }
      Z.col(t) = tmp;
      
      /* Mean of the normal proposal */ 
      tmp = (Ginv_old.slice(t)) * ((grad_old.row(t)).t());
      for (p=0 ; p<P ; p++) {
        tmp(p) *= step_sq(t);
        tmp(p) += f(t,p,d);
      }
      mean_old.col(t) = tmp;
      
      /* New value */ 
      for (p=0 ; p<P ; p++) {
        f_new(t,p) = mean_old(p,t) + Z(p,t);
      }
      
    }
    /* End proposing new values */
    
    /* Gradient and Hessian at the proposed value */
    Lf_new = f_new * L.t();
    grad_factors( Timings, w.slice(d), f_new, L, Lf_new,  crt.slice(d), grad_new, mu_new, c_xi.col(d), c_new, psi.col(d) );
    hessian_factors( G_new, Gchol_new, Ginv_new, mu_new, Timings, w.slice(d), L , c_xi.col(d), det_new, psi.col(d), flag, false );
    
    /* Kernel mean at the proposed value */
    for (t=0 ; t<nTimes ; t++) {
      tmp = (Ginv_new.slice(t)) * ((grad_new.row(t)).t());
      for (p=0 ; p<P ; p++) {
        tmp(p) *= step_sq(t);
        tmp(p) += f_new(t,p);
      }
      mean_new.col(t) = tmp;
    }
    
    /* Kernel contribution to MH ratio */
    q.fill(0.0);
    for (t=0 ; t<nTimes ; t++) {
      idx        = d*nTimes + t;
      step_sq(t) = 2.0*step(idx)*step(idx);
      tmp        = f_old.row(t).t() - mean_new.col(t);
      scalar     = (tmp.t()) * G_new.slice(t) * tmp;
      q(t)      += -0.5*log(det_new(t)) - scalar(0,0)/step_sq(t);
      tmp        = f_new.row(t).t() - mean_old.col(t);
      scalar     = (tmp.t()) * G_old.slice(t) * tmp;
      q(t)      -= -0.5*log(det_old(t)) - scalar(0,0)/step_sq(t);
    }
    
    /* Likelihood contribution to MH ratio */
    for ( t=0 ; t<nTimes ; t++ ){
      MH(t) = 0.0;
      for (i=0 ; i<nUnits ; i++) {
        if ( (w(t,i,d)>0) && (t<(Timings(i)-1)) ){
          MH(t) += w(t,i,d)*c_xi(0,d)*c_xi(1,d)*( mu_new(t,i) - mu_old(t,i) ) + crt(t,i,d)* log( mu_new(t,i)/mu_old(t,i) );
        }
      }
      for (p=0 ; p<P ; p++) {
        MH(t) += -0.5*( f_new(t,p)*f_new(t,p) - f_old(t,p)*f_old(t,p) )/psi(p,d);
      }
      MH(t) += q(t);
      MH(t) = exp(MH(t));
    } 
    
    /* Accept/reject */
    for ( t=0 ; t<nTimes ; t++ ) {
      idx = d*nTimes + t;
      u = R::runif(0.0,1.0);
      if ( u<=MH(t) && flag(t)==1 ) {
        for ( p=0 ; p<P ; p++ ){
          f(t,p,d) = f_new(t,p);
        }
        if (mcmc_idx>nBurn) {
          f_accept(window,idx) += 1;
        } else {
          f_accept(mcmc_idx%window,idx) = 1;
        }
      } else {
        if (mcmc_idx<=nBurn) {
          f_accept(mcmc_idx%window,idx) = 0;
        }
      }
    }
    
    
  } /* End of loop for outcomes */
    
}



/* NB factors: gradient */
void grad_factors(const arma::uvec &Timings, const arma::umat &w, const arma::mat &f, const arma::mat &L, 
                  const arma::mat &Lf,  const arma::umat &crt, arma::mat &grad, arma::mat &mu, const arma::vec &c_xi, 
                  arma::mat &c_panel, const arma::vec &psi)
{
  int nUnits = L.n_rows;
  int nTimes = f.n_rows;
  int P      = f.n_cols;
  grad.fill(0.0);
  mu.fill(0.0);
  c_panel.fill(0.0);
  int i, t, p;
  
  /* Constants terms */
  for (i=0 ; i<nUnits ; i++) {
    for (t=0 ; t<(Timings(i)-1) ; t++) {
      if ( w(t,i)>0 ) {
        mu(t,i)      = exp(Lf(t,i));
        c_panel(t,i) = w(t,i)*mu(t,i)*c_xi(0)*c_xi(1) + crt(t,i);
      }
    }
  }
  
  
  /* Factors likelihood contributions */ 
  for (t=0 ; t<nTimes ; t++) {
    for (i=0 ; i<nUnits ; i++) {
      if ( (w(t,i)>0) && (t<(Timings(i)-1)) ) {
        for (p=0 ; p<P ; p++) {
          grad(t,p) += L(i,p) * c_panel(t,i);
        }
      }
    }
  }
  
  /* Factors prior contribution */
  for (t=0 ; t<nTimes ; t++) {
    for (p=0 ; p<P ; p++) {
      grad(t,p) += -f(t,p)/psi(p);
    }
  }
  
}



/* NB factors: Hessian */
void hessian_factors( arma::cube &G, arma::cube &Gchol, arma::cube &Ginv, const arma::mat &mu, const arma::uvec &Timings, 
                      const arma::umat &w, const arma::mat &L , const arma::vec &c_xi, arma::vec &dets, const arma::vec &psi,
                      arma::uvec &flag, bool flag_chol )
{
  int P      = L.n_cols;
  int nUnits = L.n_rows; 
  int nTimes = w.n_rows;
  int i, t, p;
  arma::vec l_tmp(P);
  arma::mat ll(P,P);
  G.fill(0.0);
  double tmp;
  
  
  for ( t=0 ; t<nTimes ; t++ ) {
    
    /* Likelihood term */
    for ( i=0 ; i<nUnits ; i++ ) {
      if ( (w(t,i)>0) && (t<(Timings(i)-1)) ){
        for (p=0 ; p<P ; p++) {
          l_tmp(p) = L(i,p);
        }
        tmp = w(t,i) * mu(t,i) * c_xi(0) * c_xi(1); 
        ll  = tmp * l_tmp * (l_tmp.t());
        G.slice(t) -= ll;
      }
    }
    
    /* Prior term */ 
    for (p=0 ; p<P ; p++) {
      G(p,p,t) += 1.0/psi(p);
    }
    
    /* Evaluation */
    G.slice(t) = symmatu(G.slice(t));
    bool flag_operation = inv( Ginv.slice(t), G.slice(t) );
    if (flag_operation) {
      Ginv.slice(t)  = symmatu(Ginv.slice(t));
      dets(t)        = det(Ginv.slice(t));
      if ( flag_chol ) {
        flag_operation = chol( Gchol.slice(t), Ginv.slice(t), "lower" );
        if ( !flag_operation ) {
          flag(t) = 0;
        }
      }
    } else {
      flag(t) = 0;
    }
    
  }  
  
}



/* SM-MALA update of loadings */
// [[Rcpp::export]]
void smmala_loadings( arma::mat &L, const arma::uvec &Timings, const arma::mat &theta, const arma::cube &y, 
                      const arma::cube &f1, const arma::mat &sigma, const arma::cube &Lf1, const arma::ucube &n, 
                      const arma::cube &kappa, const arma::cube &omega, const arma::cube &f2, const arma::cube &Lf2, 
                      const arma::ucube &z, const arma::ucube &w, const arma::ucube &crt, const arma::vec &xi, 
                      const arma::cube &f3, const arma::cube &Lf3, const arma::vec &step, arma::umat &L_accept, 
                      int mcmc_idx, int nBurn)
{
  int P      = L.n_cols;
  int nTimes = y.n_rows;
  int nUnits = y.n_cols;
  int D1     = y.n_slices;
  int D2     = n.n_slices;
  int D3     = z.n_slices;
  int window = L_accept.n_rows - 1; 
  int d, i, p, t;
  arma::vec  det_old(nUnits),       det_new(nUnits);
  arma::mat  grad_old(P,nUnits),    grad_new(P,nUnits);
  arma::cube G_old(P,P,nUnits),     G_new(P,P,nUnits);  
  arma::cube Ginv_old(P,P,nUnits),  Ginv_new(P,P,nUnits);
  arma::cube Gchol_old(P,P,nUnits), Gchol_new(P,P,nUnits);
  /* Again, we increase RAM requirements to spare some calculations from being repeated */  
  arma::mat mean_old(P,nUnits), mean_new(P,nUnits), Z(P,nUnits);
  arma::vec tmp(P), step_sq(nUnits), q(nUnits), MH(nUnits); 
  arma::mat scalar(1,1);
  double u;
  arma::cube Lf1_new(nTimes,nUnits,D1), Lf2_new(nTimes,nUnits,D2), Lf3_new(nTimes,nUnits,D3);
  arma::cube mu3_old(nTimes,nUnits,D3), mu3_new(nTimes,nUnits,D3);
  arma::cube c_binom_old(nTimes,nUnits,D2), c_binom_new(nTimes,nUnits,D2), c_negbin(nTimes,nUnits,D3);
  arma::mat L_new(nUnits,P);
  arma::uvec flag(nUnits); flag.fill(1);
  
  
  /* Variance related-terms used multiple times */
  arma::mat c_sigma(nUnits, D1);
  for ( d=0 ; d<D1 ; d++ ){
    for ( i=0 ; i<nUnits ; i++ ) {
      c_sigma(i,d) = 1.0/sigma(i,d);
    }
  }
  
  
  /* Dispersion-related constants */
  arma::mat c_xi(2,D3);
  for ( d=0 ; d<D3 ; d++ ){
    c_xi(0,d) = 1.0/xi(d);
    c_xi(1,d) = log( 1.0/(1.0+xi(d)) );
  }
  
  
  /* Gradient and Hessian at the current value */
  grad_loadings ( grad_old, Timings, L, theta, y, c_sigma, Lf1, f1, omega, n, kappa, Lf2, c_binom_old, f2, w, z, 
                  crt, c_xi, c_negbin, Lf3, f3, mu3_old );
  hessian_loadings( det_old, G_old, Gchol_old, Ginv_old, Timings, theta, f1, c_sigma, f2, n, omega, f3, w, mu3_old, c_xi, flag, true );
  
  /* Propose new values */
  for ( i=0 ; i<nUnits ; i++ ){
    step_sq(i) = 0.5*step(i)*step(i);
    
    /* The noise of the proposal */
    for (p=0 ; p<P ; p++) {
      tmp(p) = R::rnorm(0.0,1.0);
    }
    tmp = (Gchol_old.slice(i)) * tmp;
    for (p=0 ; p<P ; p++) {
      tmp(p) *= step(i);
    }
    Z.col(i) = tmp;
    
    /* Mean of the normal proposal */ 
    tmp = Ginv_old.slice(i) * grad_old.col(i);
    for (p=0 ; p<P ; p++) {
      tmp(p) *= step_sq(i);
      tmp(p) += L(i,p);
    }
    mean_old.col(i) = tmp;
    
    /* New value */ 
    for (p=0 ; p<P ; p++) {
      L_new(i,p) = mean_old(p,i) + Z(p,i);
    }
  }
  
  /* Evaluate linear predictors at the proposed value */
  for ( d=0 ; d<D1 ; d++ ) { Lf1_new.slice(d) = f1.slice(d) * L_new.t(); }
  for ( d=0 ; d<D2 ; d++ ) { Lf2_new.slice(d) = f2.slice(d) * L_new.t(); }
  for ( d=0 ; d<D3 ; d++ ) { Lf3_new.slice(d) = f3.slice(d) * L_new.t(); }
  
  /* Gradient and Hessian at proposed value */
  grad_loadings ( grad_new, Timings, L, theta, y, c_sigma, Lf1_new, f1, omega, n, kappa, Lf2_new, c_binom_new, f2, w, z, 
                  crt, c_xi, c_negbin, Lf3_new, f3, mu3_new );
  hessian_loadings( det_new, G_new, Gchol_new, Ginv_new, Timings, theta, f1, c_sigma, f2, n, omega, f3, w, mu3_new, c_xi, flag, false );
  
  /* Kernel mean at proposed value */
  for (i=0 ; i<nUnits ; i++) {
    tmp = Ginv_new.slice(i) * grad_new.col(i);
    for (p=0 ; p<P ; p++) {
      tmp(p) *= step_sq(i);
      tmp(p) += L_new(i,p);
    }
    mean_new.col(i) = tmp;
  }
  
  /* Kernel contribution to MH ratio */
  q.fill(0.0);
  for (i=0 ; i<nUnits ; i++) {
    step_sq(i) = 2.0*step(i)*step(i);
    tmp    = L.row(i).t() - mean_new.col(i) ;
    scalar = (tmp.t()) * G_new.slice(i) * tmp;
    q(i)  += -0.5*log(det_new(i)) - scalar(0,0)/step_sq(i);
    tmp    = L_new.row(i).t() - mean_old.col(i) ;
    scalar = (tmp.t()) * G_old.slice(i) * tmp;
    q(i)  -= -0.5*log(det_old(i)) - scalar(0,0)/step_sq(i);
  }
  
  /* Likelihood contribution to MH ratio */
  for (i=0 ; i<nUnits ; i++) {
    MH(i) = 0.0;
    for (t=0 ; t<(Timings(i)-1) ; t++) {
      /* Normal outcomes */
      for (d=0 ; d<D1 ; d++) {
        MH(i) += -0.5*c_sigma(i,d)* ( pow(y(t,i,d)-Lf1_new(t,i,d),2.0) - pow(y(t,i,d)-Lf1(t,i,d),2.0) );
      }
      /* Binomial outcomes */
      for (d=0 ; d<D2 ; d++) {
        if (n(t,i,d)>0) {
          MH(i) += -0.5*omega(t,i,d)* ( pow( c_binom_new(t,i,d)-Lf2_new(t,i,d) ,2.0 ) - pow( c_binom_old(t,i,d)-Lf2(t,i,d),2.0 ) );
        }
      }
      /* NB outcomes */
      for (d=0 ; d<D3 ; d++) {
        if (w(t,i,d)>0) {
          MH(i) += w(t,i,d)*c_xi(0,d)*c_xi(1,d)*( mu3_new(t,i,d) - mu3_old(t,i,d) ) + crt(t,i,d)* log( mu3_new(t,i,d)/mu3_old(t,i,d) );
        }
      }
    }
    /* Prior contribution */
    for (p=0 ; p<P ; p++) {
      MH(i) += -0.5*( L_new(i,p)*L_new(i,p) - L(i,p)*L(i,p) )/theta(i,p);
    }
    MH(i) += q(i);
    MH(i) = exp(MH(i));
  }
  
  /* Accept/reject */
  for ( i=0 ; i<nUnits ; i++ ) {
    u = R::runif(0.0,1.0);
    if ( u<=MH(i) && flag(i)==1 ) {
      for ( p=0 ; p<P ; p++ ){
        L(i,p) = L_new(i,p);
      }
      if (mcmc_idx>nBurn) {
        L_accept(window,i) += 1;
      } else {
        L_accept(mcmc_idx%window,i) = 1;
      }
    } else {
      if (mcmc_idx<=nBurn) {
        L_accept(mcmc_idx%window,i) = 0;
      }
    }
  }
  
  
}



/* Loadings: gradient */
void grad_loadings ( arma::mat &grad, const arma::uvec &Timings, const arma::mat &L, const arma::mat &theta, 
                     const arma::cube &y, const arma::mat &c_sigma, const arma::cube &Lf1, const arma::cube &f1, 
                     const arma::cube &omega, const arma::ucube &n, const arma::cube &kappa, const arma::cube &Lf2, 
                     arma::cube &c_binom, const arma::cube &f2, const arma::ucube &w, const arma::ucube &z, 
                     const arma::ucube &crt, const arma::mat &c_xi, arma::cube &c_negbin, const arma::cube &Lf3, 
                     const arma::cube &f3, arma::cube &mu3 )
{
  int nUnits = L.n_rows;
  int P      = L.n_cols;
  int D1     = y.n_slices;
  int D2     = n.n_slices;
  int D3     = w.n_slices;
  grad.fill(0.0); c_binom.fill(0.0); c_negbin.fill(0.0); mu3.fill(0.0);
  int i, d, t, p;
  
  
  /* Binomial common terms  */
  for (d=0 ; d<D2 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<(Timings(i)-1) ; t++) {
        if (n(t,i,d)>0) {
          c_binom(t,i,d) = kappa(t,i,d)/omega(t,i,d);
        }
      }
    }
  }
  
  /* NB common terms */
  for (d=0 ; d<D3 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<(Timings(i)-1) ; t++) {
        if (w(t,i,d)>0) {
          mu3(t,i,d)      = exp(Lf3(t,i,d));
          c_negbin(t,i,d) = w(t,i,d)*mu3(t,i,d)*c_xi(0,d)*c_xi(1,d) + crt(t,i,d);
        }
      }
    }
  }
  
  /* Normal likelihood contributions */
  for (d=0 ; d<D1 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<(Timings(i)-1) ; t++) {
        for (p=0 ; p<P ; p++) {
          grad(p,i) += c_sigma(i,d) * f1(t,p,d) * ( y(t,i,d) - Lf1(t,i,d) );
        }
      }
    }
  }
  
  /* Binomial likelihood contributions */
  for (d=0 ; d<D2 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<(Timings(i)-1) ; t++) {
        if (n(t,i,d)>0) {
          for (p=0 ; p<P ; p++) {
            grad(p,i) += omega(t,i,d) * f2(t,p,d) * ( c_binom(t,i,d) - Lf2(t,i,d) );
          }
        }
      }
    }
  }
  
  /* Negative binomial likelihood contributions */
  for (d=0 ; d<D3 ; d++) {
    for (i=0 ; i<nUnits ; i++) {
      for (t=0 ; t<(Timings(i)-1) ; t++) {
        if (w(t,i,d)>0) {
          for (p=0 ; p<P ; p++) {
            grad(p,i) += f3(t,p,d) * c_negbin(t,i,d) ;
          }
        }
      }
    }
  }
  
  /* Prior contributions */
  for(i=0 ; i<nUnits ; i++) {
    for (p=0 ; p<P ; p++) {
      grad(p,i) += -L(i,p)/theta(i,p);
    }
  }
  
  /* Note: reduce length of code above by using a single loop over units */
}



/* Loadings: Hessian */
void hessian_loadings( arma::vec &dets, arma::cube &G, arma::cube &Gchol, arma::cube &Ginv, 
                       const arma::uvec &Timings, const arma::mat &theta, 
                       const arma::cube &f1, const arma::mat &c_sigma,
                       const arma::cube &f2, const arma::ucube &n, const arma::cube &omega,
                       const arma::cube &f3, const arma::ucube &w, const arma::cube &mu3, const arma::mat &c_xi,
                       arma::uvec &flag, bool flag_chol )
{
  int P      = f1.n_cols;
  int nUnits = n.n_cols;
  int D1     = f1.n_slices;
  int D2     = f2.n_slices;
  int D3     = f3.n_slices;
  int i, t, p, d;
  arma::vec f_tmp(P);
  arma::mat ff(P,P);
  G.fill(0.0);
  double tmp;
  
  /* Likelihood contributions */
  for (i=0 ; i<nUnits ; i++) {
    for (t=0 ; t<(Timings(i)-1) ; t++) {
      /* Normal outcomes */
      for (d=0 ; d<D1 ; d++) {
        for (p=0 ; p<P ; p++) {
          f_tmp(p) = f1(t,p,d);
        }
        ff = c_sigma(i,d) * f_tmp * (f_tmp.t());
        G.slice(i) += ff;
      }
      /* Binomial outcomes */
      for (d=0 ; d<D2 ; d++) {
        if (n(t,i,d)>0) {
          for (p=0 ; p<P ; p++) {
            f_tmp(p) = f2(t,p,d);
          }
          ff = omega(t,i,d) * f_tmp * (f_tmp.t());
          G.slice(i) += ff;
        }
      }
      /* Negative binomial outcomes */
      for (d=0 ; d<D3 ; d++) {
        if (w(t,i,d)>0) {
          for (p=0 ; p<P ; p++) {
            f_tmp(p) = f3(t,p,d);
          }
          tmp = w(t,i,d) * mu3(t,i,d) * c_xi(0,d) * c_xi(1,d); 
          ff  = tmp * f_tmp * (f_tmp.t());
          G.slice(i) -= ff;
        }
      }
    }
    
    /* Prior contribution */ 
    for (p=0 ; p<P ; p++) {
      G(p,p,i) += 1.0/theta(i,p);
    }
    
    /* Matrix manipulations */
    G.slice(i) = symmatu(G.slice(i));
    bool flag_operation = inv( Ginv.slice(i), G.slice(i) );
    if (flag_operation) {
      Ginv.slice(i) = symmatu(Ginv.slice(i));
      dets(i)       = det(Ginv.slice(i));
      if ( flag_chol ) {
        flag_operation = chol( Gchol.slice(i), Ginv.slice(i), "lower" );
        if ( !flag_operation ) {
          flag(i) = 0;
        }
      }
    } else {
      flag(i) = 0;
    }
    
  } /* End of loop for units */
  
}
