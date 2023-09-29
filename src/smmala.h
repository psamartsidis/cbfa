
void barker_dispersion( const arma::uvec &, const arma::ucube &, const arma::ucube &, 
                        arma::vec &, const arma::vec &, const arma::cube &, arma::umat &, int, int );
                        
void smmala_negbin_factors(const arma::mat &, arma::cube &, const arma::uvec &, const arma::vec &, 
                           const arma::ucube &, const arma::ucube &, const arma::mat &, const arma::cube &, 
                           const arma::vec &, arma::umat &, int , int );

void smmala_loadings( arma::mat &, const arma::uvec &, const arma::mat &, const arma::cube &, const arma::cube &, const arma::mat &, 
                      const arma::cube &, const arma::ucube &, const arma::cube &, const arma::cube &, const arma::cube &, const arma::cube &, 
                      const arma::ucube &, const arma::ucube &, const arma::ucube &, const arma::vec &, const arma::cube &, 
                      const arma::cube &, const arma::vec &, arma::umat &, int , int );