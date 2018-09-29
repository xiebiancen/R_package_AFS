#include <RcppArmadilloExtensions/sample.h>
#include <iostream>
#include "stat.h"

using namespace Rcpp;
using namespace std;
using namespace arma;



const double log2pi = log(2.0 * M_PI);
// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::depends(RcppArmadillo)]]
// get eigenvalues
// [[Rcpp::export]]
arma::vec getEigenValuesd(arma::mat M,arma::vec psi){

    arma::cx_vec eigval;
    eig_gen(eigval,M);
    arma::vec ret;
    ret=arma::abs(eigval);
    return ret;
}


//  log gamma function
//[[Rcpp::export]]
double lngamma(double a) {
    Rcpp::NumericVector ar = Rcpp::wrap(a);
    Rcpp::NumericVector lgr = Rcpp::lgamma(ar);
    double lgd = lgr[0];
    return lgd;
}

//log multivariate gamma function
//[[Rcpp::export]]
arma::vec lngammamv(arma::vec a) {
    Rcpp::NumericVector ar = Rcpp::wrap(a);
    Rcpp::NumericVector lgr = Rcpp::lgamma(ar);
    return lgr;
}


// log pdf of the univariate t density
// (y-m)/sqrt(tau2) is a classic student t distribution with d.f. nu
// [[Rcpp::export]]
double pdft1_(double y, double m, double tau2, double nu) {
    double nu1,nu2,cnst;
    double e,ker,z;
    nu1 = (nu+1.0)/2.0;
    nu2 = nu/2.0;
    cnst = lngamma(nu1) - lngamma(nu2) - 0.5*log(arma::datum::pi*nu);
    e = (y-m);
    ker = -nu1 * log(1.0 + e*e/(tau2*nu) );
    z = cnst - log(sqrt(tau2)) + ker;
    return z;
}



//generate random variables from standard normal distribution
//[[Rcpp::export]]
arma::colvec rnorma(int n) {
    Rcpp::RNGScope scope;
    arma::colvec a;
    Rcpp::NumericVector ar;
    ar = Rcpp::rnorm(n);
    a = Rcpp::as<colvec>(ar);
    return a;
}

//log of multivatiate t
// [[Rcpp::export]]
double pdft1mv(arma::vec y, arma::vec m, arma::mat P, int nu) {
    int n =y.n_rows;
    double a = (n+nu)*0.5;
    double cnst = lngamma(a) - lngamma(nu*0.5) - n*0.5*log(arma::datum::pi*nu);
    arma::mat C = cholmod(P);
    arma::vec e = arma::trans(C)*(y-m);
    double ePe = sum(e%e);
    double ker = -a*log(1.0 + ePe/nu);
    double z = cnst + 0.5*lndet1(C) + ker;
    return z;
}

// log multivariate t distribution
//[[Rcpp::export]]
double pdfmvt1(arma::colvec y, arma::colvec mu,
               arma::mat P, double nu) {
    int k = y.n_rows;
    double a = (k+nu)/2.0;
    double lnc = lngamma(a) - lngamma(nu/2.0) - (k/2.0)*log(math::pi()*nu);
    arma::colvec e = y-mu;
    double val = log(det(P));
    double ePe = arma::as_scalar(arma::trans(e) * P * e);
    double lpdf = lnc + 0.5*val - a*log( 1.0 + ePe/nu );
    return lpdf;
}




// log det
//[[Rcpp::export]]
double lndet1(arma::mat C){
    double ret = 2.0*sum(log(diagvec(C)));
    return ret;
}


//pdf for beta distribution
// [[Rcpp::export]]
double lnpdfbeta(double p, double a, double b){
    double retf = lngamma(a+b)-lngamma(a)-lngamma(b);
    retf=retf+(a-1)*log(p)+(b-1)*log(1-p);
    return(retf);
}


//log of the inverse gamma density function
//sig2 is the scalar (positive) value of the random variable
//v is the first parameter of the IG density
//d is the second parameter, mean is d/v
//in wikipedia, v is alpha(shape parameter) and d is beta(scale parameter)
//log of the IG density at sig2
// [[Rcpp::export]]
double pdfig1(double sig2, double v, double d) {
    double c = v * log(d) - lngamma(v);
    double lpdf = c - (v+1.0) * log(sig2) - d/sig2;
    return lpdf;
}


//generate random variable from uniform distribution
//[[Rcpp::export]]
arma::vec randitg(int m, int n){
    arma::vec temp = arma::randu(n,1);
    arma::vec temp1 = m*arma::ones<arma::vec>(n);
    arma::vec ret = temp%temp1;
    return(arma::ceil(ret));

  
}

// sample with replacement
// [[Rcpp::export]]
Rcpp::IntegerVector multisample(int nmh, int n){

    //Rcpp::RNGScope scope;
    Rcpp::IntegerVector rx = seq_len(nmh);
    double nmh1=static_cast<double>(nmh);
    NumericVector rprobm = rep(1/nmh1,nmh);
    Rcpp::IntegerVector ret = Rcpp::RcppArmadillo::sample(rx, n, false, rprobm);
    //arma::uvec ret = Rcpp::RcppArmadillo::sample(rx, n, true, rprob);
    return(ret);
}


// log pdf inverse gamma
//[[Rcpp::export]]
arma::vec pdfig1mv(arma::vec sig2, arma::vec v, arma::vec d){
    arma::vec c = v% log(d) - lngammamv(v);
    arma::vec lpdf = c - (v+1.0)%log(sig2) - d/sig2;
    return lpdf;
}

// uniform distribution
//[[Rcpp::export]]
double rndu_() {
    Rcpp::RNGScope scope;
    double a;
    Rcpp::NumericVector ar;
    ar = Rcpp::runif(1);
    a = ar[0];
    return a;
}

//univariate normal random variables
// [[Rcpp::export]]
double rndn_() {
    Rcpp::RNGScope scope;
    double a;
    Rcpp::NumericVector ar;
    ar = Rcpp::rnorm(1);
    a = ar[0];
    return a;
}

//simulates from the univariate-t distribution
//nu is the df
//m is the location parameter (mean if nu > 1)
//V is the dispersion parameter (variance if nu > 2)<- incorrect
//When nu>2, variance is V*nu/(nu-2)
//This is the non standardized student t distribution.
//In wikipedia, mu is the mu (location), V is square of
//sigma (scale parameter)
//NOTE:This is NOT the Noncentral t distribution.
//draw from the univariate-t distribution
//Siddhartha Chib
//[[Rcpp::export]]
double rta(double nu, double m, double V){

    double C = sqrt(V);
    arma::colvec ev = rnorm(1);
    double e = ev(0);
    double x = C*e;
    double y = rchisqs(nu)/nu;
    double z = m + x / sqrt(y);
    return z;
}


//random variables from inverse gamma
//[[Rcpp::export]]
arma::colvec rigamman(int n, double a, double b) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector av = Rcpp::rgamma(n,a,1.0)/b;
    av = 1.0/av;
    return av;
}

//univariate inverse gamma 
//[[Rcpp::export]]
double rigamma(double a, double b) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector av = Rcpp::rgamma(1,a,1.0)/b;
    av = 1.0/av;
    return av(0);
}

//chisquare
// [[Rcpp::export]]
double rchisqs(double nu) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector ajr = Rcpp::rchisq(1,nu);
    double aj = ajr[0];
    return aj;
}

// multivariate t
// [[Rcpp::export]]
arma::colvec rmvta(double nu, arma::colvec m, arma::mat V) {
    int k = m.n_rows;
    arma::mat C = cholmod(V);
    arma::colvec e = rnorm(k);
    arma::colvec x = C*e;
    double y = rchisqs(nu)/nu;
    arma::colvec z = m + x / sqrt(y);
    return z;
}

//[[Rcpp::export]]
Rcpp::List reparaig(double s2mean, double s2sd){
    double a, b;
    a = s2mean * s2mean / s2sd / s2sd + 2;
    b = s2mean * (a - 1.0);
    a = a * 2;
    b = b * 2;
    return Rcpp::List::create(Rcpp::Named("a") = a,
                              Rcpp::Named("b") = b);
}



// cholesck decomposition
// [[Rcpp::export]]
arma::mat cholmod(arma::mat A){
    int n = A.n_rows;
    double epsilon = 1e-16;
    arma::vec diagA = A.diag();
    double gamma = max(abs(diagA));
    double xi = max(max(abs(A-diagmat(diagA))));
    arma::vec k1 = ones<arma::vec>(2);
    k1(0) = gamma + xi;
    double delta = epsilon*max(k1);
    arma::vec k2 = zeros<arma::vec>(3);
    k2(0) = gamma;
    k2(1) = xi/n;
    k2(2) = epsilon;
    double beta = sqrt(max(k2));
    //    int indef = 0;

    arma::rowvec d = arma::zeros<arma::rowvec>(n);
    arma::mat L = eye<arma::mat>(n,n);
    for (int j = 0;j<n;++j){
        double djtemp, theta;
        if(j==0){
            djtemp = A(j,j);
        } else{
            djtemp = A(j,j)-sum(d(span(0,j-1))%L(j,span(0,j-1))%L(j,span(0,j-1)));
        }
        arma::colvec k4 = zeros<arma::colvec>(2);
        k4(0) = abs(djtemp);
        k4(1) = delta;
        double k5 = max(k4);
        if(j<n-1){
            arma::colvec Ccol;
            if(j==0){
                Ccol = A(span(j+1,n-1),j);
            } else{
                Ccol = A(span(j+1,n-1),j)-L(span(j+1,n-1),span(0,j-1))*arma::trans(d(span(0,j-1))%L(j,span(0,j-1)));
            }
            theta = max(abs(Ccol));
            arma::colvec k3 = zeros<arma::colvec>(2);
            k3(0) = k5;
            k3(1) = (theta/beta)*(theta/beta);
            double dj = max(k3);
            d(j) = dj;
            L(span(j+1,n-1),j) = Ccol/dj;
        } else{
            d(j) = k5;
        }
        //        if(arma::as_scalar(d(j))>djtemp){
        //            indef = 1;
        //        }
    }
    L = L*diagmat(sqrt(d));
    //    Amod = L*arma::trans(L);
    return(L);
    //    return(Amod);
}

//inverse of upper triangle
//[[Rcpp::export]]
arma::mat invuptr(arma::mat T){

    int m = T.n_rows;
    int n = T.n_cols;
    if(m!=n){
        Rcout<<"matrix is not square"<<endl;
    }

    for(int i = n-1; i>=0;i--){

        if(T(i,i)==0){
            Rcout<<"matrix is singular"<<endl;
        }
        T(i,i)=1/T(i,i);

        for(int j=i-1;j>=0;j--){
            double sum = 0;
            double e = arma::sum(T(j,span(j+1,i))*T(span(j+1,i),i));
            sum = sum +e;
            T(j,i)=-sum/T(j,j);
        }
    }
    return(T);
}

//inverse of lower triangle
//[[Rcpp::export]]
arma::mat invlptr(arma::mat A){

    arma::mat B = arma::trans(A);
    arma::mat T = invuptr(B);
    return(arma::trans(T));
}


//inverse for positive definite
//[[Rcpp::export]]
arma::mat invpd(arma::mat A){

    arma::mat H = arma::trans(cholmod(A));
    //arma::mat H = cholmod(A);
    //arma::mat temp =H*arma::trans(H);
    arma::mat Hinv = invuptr(H);
    arma::mat Ainv = Hinv*arma::trans(Hinv);
    //arma::mat Ainv=arma::inv(temp);
    Ainv = (Ainv+arma::trans(Ainv))/2;
    return(Ainv);
}

//column mean
//[[Rcpp::export]]
arma::vec col_means(arma::mat A){
    int nCols = A.n_cols;
    arma::vec out = arma::zeros<arma::vec>(nCols);
    for(int i = 0; i<nCols; i++){
        arma::vec tmp = A.col(i);
        out(i) = arma::mean(tmp);
    }
    return out;
}

// row mean
//[[Rcpp::export]]
arma::vec row_means(arma::mat A){
    int nRows = A.n_rows;
    arma::vec out = arma::zeros<arma::vec>(nRows);
    for(int i = 0; i<nRows; i++){
        arma::vec tmp = arma::trans(A.row(i));
        out(i) = arma::mean(tmp);
    }
    return out;
}

//log pdf normal distribution
// [[Rcpp::export]]
double pdflogmvn(arma::vec x,
                 arma::vec mean,
                 arma::mat sigma) {
    int xdim = x.n_rows;
    double out;
    arma::mat rooti = invlptr(cholmod(sigma));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;

    arma::vec z = rooti * ( x - mean ) ;
    out = constants - 0.5 * arma::sum(z%z) + rootisum;
    return(out);
}




//log pdf mean
//[[Rcpp::export]]
double pdfavg(arma::vec pdfthm){
    double thmax = max(pdfthm);
    //double thmax=0.0;
    arma::vec pdfthm1 = exp(pdfthm-thmax*arma::ones<arma::vec>(pdfthm.n_rows));
    double pdfth = log(mean(pdfthm1.elem(find_finite(pdfthm1))))+thmax;
    //double pdfth=log(mean(pdfthm1));
    return(pdfth);
}

// parameter transformation
//[[Rcpp::export]]
arma::vec ParTran(arma::vec par, arma::vec lb, arma::vec ub, double chi, int con_unc){

    int dim = par.n_rows;
    arma::vec par1=arma::zeros<arma::vec>(dim);
    if(con_unc==1){
        for(int k = 0; k<dim;k++){
            if((arma::is_finite(lb.row(k)))&(~arma::is_finite(ub.row(k)))){
                par1.row(k)=lb.row(k)+exp(chi*par.row(k));
            }else if((~arma::is_finite(lb.row(k)))&(arma::is_finite(ub.row(k)))){
                par1.row(k)=ub.row(k)-exp(chi*par.row(k));
            }else{
                par1.row(k)=lb.row(k)+(ub.row(k)-lb.row(k))/(1+exp(chi*par.row(k)));
            }
        }
    }
    else{
        for(int k = 0; k<dim;k++){
            if((arma::is_finite(lb.row(k)))&(~arma::is_finite(ub.row(k)))){

                par1.row(k)=log(par.row(k)-lb.row(k))/chi;
            }else if((~arma::is_finite(lb.row(k)))&(arma::is_finite(ub.row(k)))){

                par1.row(k)=log(ub.row(k)-par.row(k))/chi;
            }else{
                par1.row(k)=log((ub.row(k)-lb.row(k))/(par.row(k)-lb.row(k))-1)/chi;
            }
        }
    }
    return par1;
}
