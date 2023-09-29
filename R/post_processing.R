fitted_values = function( id, fit, Timings, y, n, k, z, w ) {
  
  
  # Dataset details
  P      = dim(fit$L)[2]
  D1     = dim(y)[3]
  D2     = dim(k)[3]
  D3     = dim(z)[3]
  nDraws = dim(fit$L)[3]
  nTimes = dim(y)[1]
  
  
  # 
  fit_normal     = alpha                                      = array( NA, c(nDraws,nTimes,D1) )
  fit_binom_prob = fit_binom_counts = beta_prob = beta_counts = array( NA, c(nDraws,nTimes,D2) )
  fit_negbin     = delta                                      = array( NA, c(nDraws,nTimes,D3) )
  
  
  # Normal outcomes
  if ( D1>0 ) {
    for ( d in 1:D1 ) {
      
      # indices for the factors
      idx = 1:P + (d-1)*P
      
      # Fitted values 
      for ( b in 1:nDraws ) {
        lin_pred         = fit$f_normal[,c(idx),b] %*% fit$L[id,,b]
        fit_normal[b,,d] = lin_pred + rnorm( nTimes, 0, sqrt( fit$sigma[id,1,b] ) )
      }
      
      # Causal effects
      if ( Timings[id]<=nTimes ) {
        alpha[ , ,d ]                  = matrix( y[,id,d], nDraws, nTimes, byrow=T ) - fit_normal[,,d]
        alpha[ , 1:(Timings[id]-1), d ] = 0  
      } 
    }
  }
  
  
  # Binomial outcomes
  if ( D2>0 ) {
    for ( d in 1:D2 ) {
      
      # indices for the factors
      idx = 1:P + (d-1)*P
      
      # Fitted values 
      for ( b in 1:nDraws ) {
        lin_pred               = exp( fit$f_binom[,c(idx),b] %*% fit$L[id,,b] )
        lin_pred               = lin_pred / (1+lin_pred)
        fit_binom_prob[b,,d]   = lin_pred
        fit_binom_counts[b,,d] = rbinom( nTimes, n[,id,d], lin_pred )
      }
      
      # Causal effects
      if ( Timings[id]<=nTimes ) {
        # Effect on probability
        beta_shape                          = 1 + rep( k[,id,d], times=nDraws )
        beta_rate                           = 1 + rep( n[,id,d], times=nDraws ) - rep( k[,id,d], times=nDraws )
        beta_draw                           = matrix( rbeta( nTimes*nDraws, beta_shape, beta_rate), nDraws, nTimes, byrow=T )
        beta_prob[ , ,d ]                   = beta_draw- fit_binom_prob[,,d]
        beta_prob[ , 1:(Timings[id]-1), d ] = 0
        # Separable effect on counts
        beta_counts[ , ,d ]                   = matrix( k[,id,d], nDraws, nTimes, byrow=T ) - fit_binom_counts[,,d]
        beta_counts[ , 1:(Timings[id]-1), d ] = 0  
      } 
    }
  }
  
  
  # Count outcomes
  if ( D3>0 ) {
    for ( d in 1:D3 ) {
      
      # indices for the factors
      idx = 1:P + (d-1)*P
      
      # Fitted values 
      for ( b in 1:nDraws ) {
        lin_pred         = exp( fit$f_negbin[,c(idx),b] %*% fit$L[id,,b] )
        nb_prob          = 1 / (1 + fit$xi[b,d] )
        nb_size          = w[,id,d] * lin_pred / fit$xi[b,d]
        fit_negbin[b,,d] = rnbinom( nTimes, nb_size, nb_prob )
      }
      
      # Causal effects
      if ( Timings[id]<=nTimes ) {
        delta[ , ,d ]                   = matrix( z[,id,d], nDraws, nTimes, byrow=T ) - fit_negbin[,,d]
        delta[ , 1:(Timings[id]-1), d ] = 0  
      } 
    }
  }
  
  
  # Output
  out = list( fit_normal       = fit_normal, 
              alpha            = alpha,
              fit_binom_prob   = fit_binom_prob,
              fit_binom_counts = fit_binom_counts,
              beta_prob        = beta_prob,
              beta_counts      = beta_counts, 
              fit_negbin       = fit_negbin,
              delta            = delta
              )
  return(out)
}