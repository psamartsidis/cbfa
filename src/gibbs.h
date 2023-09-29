

void pg_gibbs( const arma::ucube &, const arma::cube &, arma::cube &, const arma::uvec & );

void crt_gibbs( const arma::vec &, const arma::uvec &, const arma::ucube &, const arma::ucube &, const arma::cube &, arma::ucube & );

void f_binom_gibbs( const arma::mat &, arma::cube &, const arma::cube &, const arma::mat &, 
                    const arma::cube &, const arma::uvec &, const arma::uvec &);

void f_normal_gibbs( const arma::uvec &, const arma::uvec &, const arma::cube &, const arma::mat &, const arma::mat &, 
                     const arma::mat &, arma::cube & );

void sigma_gibbs( const arma::cube &, const arma::cube &, const arma::uvec &, arma::mat &, double );

void psi_gibbs( const arma::cube &, const arma::cube &, const arma::cube &, arma::mat &, arma::mat &, arma::mat & );

void psi_gibbs_latent( const arma::cube &, const arma::cube &, const arma::cube &, arma::mat &, arma::mat &, arma::mat & );

arma::vec tpb_gibbs( const arma::mat &, arma::mat &, arma::mat &, arma::vec &, arma::vec &, 
                     double , double , const arma::vec & );
