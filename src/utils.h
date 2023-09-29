
arma::vec rueMVnorm( const arma::vec &, const arma::mat &, const arma::vec & );

arma::umat offset_save( const arma::ucube & , const arma::uvec & );

void save_state( int , const arma::umat &, const arma::cube &, arma::mat &, const arma::vec &, arma::mat &, 
                 const arma::umat &, const arma::ucube &, arma::umat &, const arma::cube &, arma::cube &, 
                 const arma::cube &, arma::cube &, const arma::cube &, arma::cube &, const arma::mat &, arma::cube &,
                 const arma::mat &, const arma::mat &, const arma::mat &, arma::cube &, const arma::mat &, arma::cube &,
                 const arma::vec &, arma::mat &, const arma::vec &, arma::mat &, double, arma::vec &, double , arma::vec &);

void update_Lf( const arma::mat &, const arma::cube &, const arma::cube &, const arma::cube &, 
                arma::cube &, arma::cube &, arma::cube & );

void update_stepsize (int , int , double , double , const arma::umat &, arma::vec &, arma::vec & );

void acceptance_ratios( int , int , const arma::umat &, arma::vec &,const arma::umat &, arma::mat &, const arma::umat &, arma::vec & ); 
