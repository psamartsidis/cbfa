// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-


// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "mcmc.h"
#include "gibbs.h"
#include "utils.h"
#include "smmala.h"


// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Rcpp::List cbfa( int nMCMC, int nBurn, int nThin, int window, int P, 
                 const arma::cube &y, const arma::ucube &n, const arma::cube &k, 
                 const arma::ucube &w, const arma::ucube &z,const arma::uvec &Timings, 
                 double SMAX, const arma::vec &tpb_prior, int tpb_nBurn) 
{
  
  
  /* Dataset parameters */ 
  int nTimes = n.n_rows;
  int nUnits = n.n_cols;
  int D1     = y.n_slices;  /* # continuous outcomes */
  int D2     = n.n_slices;  /* # binomial outcomes */ 
  int D3     = w.n_slices;  /* # count outcomes */ 
  
  
  /* Some other helpful variables */
  int t, d, i, p;
  int nSaves = nMCMC/nThin;

  
  /* Loadings parameters */ 
  arma::mat L(nUnits,P,fill::randn); 
  L *= 0.001;
  arma::cube L_mcmc(nUnits,P,nSaves);
  arma::vec  L_step(nUnits);                 L_step.fill(0.0001);
  arma::umat L_accept(window+1,nUnits);      L_accept.fill(0);
  arma::vec  L_avg(nUnits);                  L_avg.fill(0.0);
  
  
  /* Loadings shrinkage parameters */
  double gamma = R::rgamma( tpb_prior(5), 1.0/tpb_prior(6) );
  double eta   = R::rgamma( tpb_prior(4), 1.0/gamma );
  arma::vec tau(P);
  arma::vec phi(P);
  for (p=0 ; p<P ; p++) {
    tau(p) = R::rgamma( tpb_prior(3), 1.0/eta );
    phi(p) = R::rgamma( tpb_prior(2), 1.0/tau(p));
  }
  arma::mat delta(nUnits,P,fill::value(1.0));
  arma::mat theta(nUnits,P,fill::value(1.0));
  /* We do not draw delta and theta from their priors to avoid extreme values */
  arma::vec gamma_mcmc(nSaves);
  arma::vec eta_mcmc(nSaves);
  arma::mat tau_mcmc(P,nSaves);
  arma::mat phi_mcmc(P,nSaves);
  arma::vec tpb_draw(2);
  
  
  /* Normal outcome variance */
  arma::mat sigma(nUnits,D1);
  sigma.fill(1.0);
  arma::cube sigma_mcmc(nUnits,D1,nSaves);
  
  
  /* Normal outcome factors */
  arma::cube f_normal(nTimes,P,D1,fill::randn);
  f_normal *= 0.001;
  arma::cube f_normal_mcmc(nTimes,P*D1,nSaves);

  
  /* Binomial outcome factors */
  arma::cube f_binom(nTimes,P,D2,fill::randn);
  f_binom *= 0.001;
  arma::cube f_binom_mcmc(nTimes,P*D2,nSaves);

  
  /* NB outcome factors */
  arma::cube f_negbin(nTimes,P,D3,fill::randn);
  f_negbin *= 0.001;
  arma::cube f_negbin_mcmc(nTimes,P*D3,nSaves);
  arma::vec  f_negbin_step(nTimes*D3);                 f_negbin_step.fill(0.0001);
  arma::umat f_negbin_accept(window+1,nTimes*D3);      f_negbin_accept.fill(0);
  arma::vec  f_negbin_avg(nTimes*D3);                  f_negbin_avg.fill(0.0);

  
  /* Loadings-factors product */
  arma::cube Lf_normal(nTimes,nUnits,D1);
  arma::cube Lf_binom(nTimes,nUnits,D2);
  arma::cube Lf_negbin(nTimes,nUnits,D3);
  update_Lf( L, f_normal, f_binom, f_negbin, Lf_normal, Lf_binom, Lf_negbin);
  
  
  /* Factors variance parameters */ 
  arma::mat psi_normal(P,D1,fill::value(1.0)); 
  arma::mat psi_binom( P,D2, fill::value(1.0));
  arma::mat psi_negbin(P,D3,fill::value(1.0));
  arma::cube psi_mcmc(P,D1+D2+D3,nSaves);
  
  
  /* PG latent variables */ 
  arma::cube omega(nTimes,nUnits,D2);
  omega.fill(0.0); 
  arma::umat omega_idx = offset_save(n,Timings);
  int omega_n = omega_idx.n_rows;
  arma::mat omega_mcmc(omega_n,nSaves);
  arma::cube kappa(nTimes,nUnits,D2);
  kappa.fill(0.0);
  for ( d=0 ; d<D2 ; d++ ) {
    for ( i=0 ; i<nUnits ; i++ ){
      for ( t=0 ; t<nTimes ; t++ ){
        kappa(t,i,d) = k(t,i,d) - 0.5*n(t,i,d);
      }
    }
  }
  
  
  /* NB dispersion parameters */ 
  arma::vec  xi(D3);                   xi.fill(0.01);
  arma::mat  xi_mcmc(D3,nSaves);
  arma::vec  xi_step(D3);              xi_step.fill(0.0001);
  arma::umat xi_accept(window+1,D3);   xi_accept.fill(0);
  arma::vec  xi_avg(D3);               xi_avg.fill(0.0);
  
  
  /* CRT latent variables */
  arma::ucube crt(nTimes,nUnits,D3);
  crt.fill(0); 
  arma::umat crt_idx = offset_save(w,Timings);
  int crt_n = crt_idx.n_rows;
  arma::umat crt_mcmc(crt_n,nSaves);
  
  
  /* # controls at each time point */ 
  arma::uvec nControls(nTimes);
  for (t=0 ; t<nTimes ; t++) {
    nControls(t) = 0;
    for (i=0 ; i<nUnits ; i++) {
      if ( (t+1)<Timings(i) ) {
        nControls(t) += 1;
      }
    }
  }
  
  
  /* ************************************************* */
  /* ************************************************* */
  /* MCMC loop */ 
  int idx=0;
  for ( int b=1 ; b<=nMCMC ; b++) {
    
    /* NB dispersion parameter */
    barker_dispersion( Timings, z, w, xi, xi_step, Lf_negbin, xi_accept, b, nBurn );
    if ( b<=nBurn) {
      update_stepsize ( b, nBurn, 0.45, 0.55, xi_accept, xi_step, xi_avg );
    }
    
    /* CRT latent variables */
    crt_gibbs( xi, Timings, w, z, Lf_negbin, crt );
    
    /* PG latent variables every 5 iterations */ 
    pg_gibbs( n, Lf_binom, omega, Timings );
    
    /* Loadings */
    smmala_loadings( L, Timings, theta, y, f_normal, sigma, Lf_normal, n, kappa, omega, f_binom, Lf_binom, 
                     z, w, crt, xi, f_negbin, Lf_negbin, L_step, L_accept, b, nBurn );
    if ( b<=nBurn) {
      update_stepsize ( b, nBurn, 0.70, 0.80, L_accept, L_step, L_avg );
    }
    
    /* Loadings/factors product */
    update_Lf( L, f_normal, f_binom, f_negbin, Lf_normal, Lf_binom, Lf_negbin);
    
    /* Loadings shrinkage parameters */
    /* Allowing some iterations so that variability is not absorbed by variance/dispersion */
    if (b>=tpb_nBurn) {
      tpb_draw = tpb_gibbs( L, theta, delta, phi, tau, eta, gamma, tpb_prior );
      eta      = tpb_draw(0);
      gamma    = tpb_draw(1);
    }
    
    /* Normal variance parameters */
    sigma_gibbs( y, Lf_normal, Timings, sigma, SMAX );

    /* Binomial factors */ 
    f_binom_gibbs( L, f_binom, kappa, psi_binom, omega, Timings, nControls);
  
    /* NB factors */ 
    smmala_negbin_factors( L, f_negbin, Timings, xi, w, crt, psi_negbin, Lf_negbin, f_negbin_step, f_negbin_accept, b, nBurn );
    if ( b<=nBurn) {
      update_stepsize ( b, nBurn, 0.70, 0.80, f_negbin_accept, f_negbin_step, f_negbin_avg );
    }
  
    /* Normal factors */
    f_normal_gibbs( nControls, Timings, y, L, sigma, psi_normal, f_normal );
    
    /* Loadings/factors product to use for the update of Loadings */
    update_Lf( L, f_normal, f_binom, f_negbin, Lf_normal, Lf_binom, Lf_negbin);
    
    /* Factors variance parameters */
    psi_gibbs_latent( f_normal, f_binom, f_negbin, psi_normal, psi_binom, psi_negbin );

    /* Save the current state of parameters */
    if (b%nThin==0) {
      Rcpp::checkUserInterrupt();
      save_state( idx, omega_idx, omega, omega_mcmc, xi, xi_mcmc, crt_idx, crt, crt_mcmc,
                  f_binom, f_binom_mcmc, f_negbin, f_negbin_mcmc, f_normal, f_normal_mcmc, sigma, sigma_mcmc,
                  psi_normal, psi_binom, psi_negbin, psi_mcmc, L, L_mcmc, tau, tau_mcmc, phi, phi_mcmc, eta, 
                  eta_mcmc, gamma, gamma_mcmc );
      idx += 1;
    }

    /* Print progress */
    if (b%1000==0) {
      Rcout << "MCMC iteration " << b << "\n";
      R_FlushConsole();
    }
    
  }
  /* ************************************************* */
  /* ************************************************* */
  
  
  /* Calculate the acceptance ratios */
  arma::vec xi_ar(D3);
  arma::mat f_negbin_ar(D3,nTimes);
  arma::vec L_ar(nUnits);
  acceptance_ratios( nMCMC, nBurn, xi_accept, xi_ar, f_negbin_accept, f_negbin_ar, L_accept, L_ar );
  
  
  /* Output */ 
  return Rcpp::List::create( Rcpp::Named("omega")           = omega_mcmc.t(),
                             Rcpp::Named("omega_idx")       = omega_idx,
                             Rcpp::Named("xi")              = xi_mcmc.t(),
                             Rcpp::Named("crt")             = crt_mcmc.t(),
                             Rcpp::Named("crt_idx")         = crt_idx,
                             Rcpp::Named("xi_accept")       = xi_ar,
                             Rcpp::Named("f_binom")         = f_binom_mcmc,
                             Rcpp::Named("f_negbin")        = f_negbin_mcmc,
                             Rcpp::Named("f_negbin_accept") = f_negbin_ar,
                             Rcpp::Named("f_normal")        = f_normal_mcmc,
                             Rcpp::Named("sigma")           = sigma_mcmc,
                             Rcpp::Named("psi")             = psi_mcmc,
                             Rcpp::Named("L")               = L_mcmc,
                             Rcpp::Named("L_accept")        = L_ar,
                             Rcpp::Named("phi")             = phi_mcmc.t(),
                             Rcpp::Named("tau")             = tau_mcmc.t(),
                             Rcpp::Named("gamma")           = gamma_mcmc,
                             Rcpp::Named("eta")             = eta_mcmc
                             );
}

