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


// [[Rcpp::export]]
double pdflogmvn1(arma::vec x,
                  arma::vec mean,
                  arma::mat sigma) {
    int xdim = x.n_rows;
    double out;
    arma::mat temp=trimatu(arma::trans(cholmod(sigma)));
    arma::mat tempinv;
    if(arma::inv(tempinv,temp)==false){
        tempinv=invuptr(temp);
    }
    arma::mat rooti = arma::trans(tempinv);
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;

    arma::vec z = rooti * ( x - mean ) ;
    out = constants - 0.5 * arma::sum(z%z) + rootisum;
    if(arma::is_finite(out)==0){
        out=-exp(20);
    }
    return(out);
}

// [[Rcpp::export]]
arma::vec getEigenValuesd(arma::mat M,arma::vec psi){

    arma::cx_vec eigval;
    eig_gen(eigval,M);
    arma::vec ret;
    if(eig_gen(eigval,M)== false){
        Rcout<<psi<<endl;
    }
    else{
        ret=arma::abs(eigval);
    }
    return ret;
}



//[[Rcpp::export]]
double lngamma(double a) {
    Rcpp::NumericVector ar = Rcpp::wrap(a);
    Rcpp::NumericVector lgr = Rcpp::lgamma(ar);
    double lgd = lgr[0];
    return lgd;
}

//[[Rcpp::export]]
arma::vec lngammamv(arma::vec a) {
    Rcpp::NumericVector ar = Rcpp::wrap(a);
    Rcpp::NumericVector lgr = Rcpp::lgamma(ar);
    return lgr;
}


// log of the univariate t density
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
    if(arma::is_finite(z)==0){
        z=-exp(20);
    }
    return z;
}

//Siddhartha Chib
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



//[[Rcpp::export]]
double dwish(arma::mat W, double nu, arma::mat Rinv){
    int k = W.n_rows;
    double c1 = -(nu*k/2)*log(2);
    double c2 = -(k*(k-1)/4)*log(arma::datum::pi);
    arma::vec nuvec1=arma::zeros<arma::vec>(k);
    for(int i=0;i<k;i++){
        nuvec1(i)=(nu-i)/2;
    }
    double c3 = -sum(lngammamv(nuvec1));
    double log_detRinv;
    double sign;
    arma::log_det(log_detRinv,sign,Rinv);
    double c4 = (nu/2)*log_detRinv;
    double log_detW;
    double sign1;
    arma::log_det(log_detW,sign1,W);
    double c5 = (nu-k-1)/2*log_detW;
    double c6 = -0.5*sum(arma::diagvec(Rinv*W));
    double lpdf = c1+c2+c3+c4+c5+c6;
    return(lpdf);
}


//[[Rcpp::export]]
double dinwish(arma::mat W, double nu, arma::mat R){
    int k = W.n_rows;
    double c1 = -(nu*k/2)*log(2);
    double c2 = -(k*(k-1)/4)*log(arma::datum::pi);
    arma::vec nuvec1=arma::zeros<arma::vec>(k);
    for(int i=0;i<k;i++){
        nuvec1(i)=(nu-i)/2;
    }
    double c3 = -sum(lngammamv(nuvec1));
    double log_detR;
    double sign;
    arma::log_det(log_detR,sign,R);
    double c4 = (nu/2)*log_detR;
    double log_detW;
    double sign1;
    arma::log_det(log_detW,sign1,W);
    double c5 = -(nu+k+1)/2*log_detW;
    double c6 = -0.5*sum(arma::diagvec(R*arma::inv(W)));
    double lpdf = c1+c2+c3+c4+c5+c6;
    return(lpdf);
}



//[[Rcpp::export]]
double lndet1(arma::mat C){
    double ret = 2.0*sum(log(diagvec(C)));
    return ret;
}

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
//Siddhartha Chib
// [[Rcpp::export]]
double pdfig1(double sig2, double v, double d) {
    double c = v * log(d) - lngamma(v);
    double lpdf = c - (v+1.0) * log(sig2) - d/sig2;
    return lpdf;
}


// [[Rcpp::export]]
int equalsample_(int n) {
    Rcpp::RNGScope scope;
    Rcpp::IntegerVector rx = seq_len(n);
    Rcpp::NumericVector rprob = Rcpp::NumericVector::create();
    Rcpp::IntegerVector ret = Rcpp::RcppArmadillo::sample(rx, 1, true, rprob);
    int a;
    a = ret[0];
    //a = rx[0];
    return a;

}

//[[Rcpp::export]]
arma::vec randitg(int m, int n){
    arma::vec temp = arma::randu(n,1);
    arma::vec temp1 = m*arma::ones<arma::vec>(n);
    arma::vec ret = temp%temp1;
    return(arma::ceil(ret));

}
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


//[[Rcpp::export]]
arma::vec pdfig1mv(arma::vec sig2, arma::vec v, arma::vec d){
    arma::vec c = v% log(d) - lngammamv(v);
    arma::vec lpdf = c - (v+1.0)%log(sig2) - d/sig2;
    return lpdf;
}

//[[Rcpp::export]]
double rndu_() {
    Rcpp::RNGScope scope;
    double a;
    Rcpp::NumericVector ar;
    ar = Rcpp::runif(1);
    a = ar[0];
    return a;
}

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

//[[Rcpp::export]]
arma::colvec rigamman(int n, double a, double b) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector av = Rcpp::rgamma(n,a,1.0)/b;
    av = 1.0/av;
    return av;
}

//[[Rcpp::export]]
double rigamma(double a, double b) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector av = Rcpp::rgamma(1,a,1.0)/b;
    av = 1.0/av;
    return av(0);
}

// [[Rcpp::export]]
double rchisqs(double nu) {
    Rcpp::RNGScope scope;
    Rcpp::NumericVector ajr = Rcpp::rchisq(1,nu);
    double aj = ajr[0];
    return aj;
}

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


//[[Rcpp::export]]
arma::mat test(arma::mat A){
    arma::mat B = trimatu(A);
    return B;
}


//[[Rcpp::export]]
arma::mat rwish(double nu,arma::mat R){
    int p = R.n_rows;
    arma::mat T = arma::zeros<arma::mat>(p,p);
    int i = 0;
    while(i<p){
        int j = 0;
        while(j<=i){
            if(i==j){
                T(i,j)=sqrt(rchisqs(nu-i+1));

            }
            else{
                T(i,j)=rndn_();
            }
            j=j+1;
        }
        i=i+1;
    }
    arma::mat A = T*arma::trans(T);
    arma::mat C = cholmod(R);
    arma::mat W=arma::trans(C)*A*C;
    return(W);
    }



//[[Rcpp::export]]
double pdfavg(arma::vec pdfthm){
    double thmax = max(pdfthm);
    //double thmax=0.0;
    arma::vec pdfthm1 = exp(pdfthm-thmax*arma::ones<arma::vec>(pdfthm.n_rows));
    double pdfth = log(mean(pdfthm1.elem(find_finite(pdfthm1))))+thmax;
    //double pdfth=log(mean(pdfthm1));
    return(pdfth);
}





// [[Rcpp::export]]
arma::uvec std_setdiff(arma::uvec& x, arma::uvec& y) {

    std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
    std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
    std::vector<int> out;

    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(out, out.end()));

    return arma::conv_to<arma::uvec>::from(out);
}


//[[Rcpp::export]]
Rcpp::List mvt_sub(arma::uvec indbj, arma::vec x2, arma::vec mu,arma::mat Sigma,double v,List Spec){
    arma::uvec indRB = Spec["indRB"];
    int dim = indRB.n_rows;
    double v1 = v;
    arma::mat Sigma1;
    arma::mat Sigma22;
    arma::vec mu1;
    //arma::uvec temp=arma::regspace<arma::uvec>(1,1,dim);
    arma::uvec indbj2 = std_setdiff(indRB,indbj);
    if(indbj.n_rows+indbj2.n_rows>dim){
        Rcout<<"parameter index incorrectly specified"<<endl;
    }

    if(indbj.is_empty()==true){
        arma::vec mu1 = mu.rows(indbj-1);
        Sigma1 = Sigma.submat(indbj-1,indbj-1);
        v1 = v;
    }else{
        arma::mat Sigma11= Sigma.submat(indbj-1,indbj-1);
        arma::mat Sigma12= Sigma.submat(indbj-1,indbj2-1);
        Sigma22=Sigma.submat(indbj2-1,indbj2-1);
        mu1 = mu.rows(indbj-1)+arma::trans(arma::trans(x2-mu.rows(indbj2-1))*arma::inv(Sigma22)*arma::trans(Sigma12));
        //Sigma1 = Sigma11-Sigma12*arma::inv(Sigma22)*arma::trans(Sigma12);
        //v1=v+indbj2.n_rows;
        //arma::mat temp3=arma::trans(x2-mu.rows(indbj2-1))*arma::inv(Sigma22)*(x2-mu.rows(indbj2-1));
        //Sigma1=(temp3(0)+v)/v1*Sigma1;
    }



    return Rcpp::List::create(Rcpp::Named("mu1") = mu1);
}

//[[Rcpp::export]]
arma::mat bfgsi(arma::mat H0,arma::vec dg,arma::vec dx){
    arma::mat Hdg = H0*dg;
    arma::mat temp = arma::trans(dg)*dx;
    arma::mat  temp1=arma::trans(dg)*Hdg;
    arma::mat H;

    double dgdx = temp(0);
    double dgHdg = temp1(0);
    if(abs(dgdx)>1e-12){
        H=H0+(1+dgHdg/dgdx)*(dx*arma::trans(dx))/dgdx-(dx*arma::trans(Hdg)+Hdg*arma::trans(dx))/dgdx;
    }
    else{
        H=H0;
    }
    return H;
}


//[[Rcpp::export]]
arma::mat rnormmat(int m,int n){
    arma::mat A = arma::zeros<arma::mat>(m,n);
    for(int i=0;i<m;i++){

        A.row(i)=as<arma::rowvec>(rnorm(n));
    }

    return(A);
}

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
