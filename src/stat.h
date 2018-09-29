double pdflogmvn(arma::vec x,
                 arma::vec mean,
                 arma::mat sigma);
double pdft1_(double y, double m, double tau2, double nu);
double pdfig1(double sig2, double v, double d);
double lngamma(double a);
double lnpdfbeta(double p, double a, double b);
int equalsample_(int n);
arma::mat cholmod(arma::mat A);
arma::colvec rmvta(double nu, arma::colvec m, arma::mat V);
double rchisqs(double nu);
double rndu_();
double rndn_();
double rta(double nu, double m, double V);
double pdft1mv(arma::vec y, arma::vec m, arma::mat P, int nu);
Rcpp::List reparaig(double s2mean, double s2sd);
double rigamma(double a, double b);
arma::mat invuptr(arma::mat T);
arma::mat invpd(arma::mat A);
arma::vec col_means(arma::mat A);
arma::vec row_means(arma::mat A);
arma::vec pdfig1mv(arma::vec sig2, arma::vec v, arma::vec d);
Rcpp::IntegerVector multisample(int nmh, int n);
double lndet1(arma::mat C);
arma::mat invlptr(arma::mat A);
double dwish(arma::mat W, double nu, arma::mat Rinv);
arma::mat rwish(double nu,arma::mat R);
double dinwish(arma::mat W, double nu, arma::mat R);
double pdfavg(arma::vec pdfthm);
arma::vec randitg(int m, int n);
arma::mat bfgsi(arma::mat H0,arma::vec dg,arma::vec dx);
arma::uvec std_setdiff(arma::uvec& x, arma::uvec& y);
Rcpp::List mvt_sub(arma::uvec indbj, arma::vec x2, arma::vec mu,arma::mat Sigma,double v,Rcpp::List Spec);
arma::vec ParTran(arma::vec par, arma::vec lb, arma::vec ub, double chi, int con_unc);
arma::mat rnormmat(int m,int n);
double crossprod(arma::vec a,arma::vec b);
arma::colvec rnorma(int n);
double pdfmvt1(arma::colvec y, arma::colvec mu, arma::mat P, double nu);
arma::vec getEigenValuesd(arma::mat M,arma::vec psi);