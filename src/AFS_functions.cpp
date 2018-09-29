#include <RcppArmadillo.h>
#include "stat.h"
using namespace Rcpp;
using namespace std;


//[[Rcpp::depends(RcppArmadillo)]]

//parameter transformation
//[[Rcpp::export]]
arma::vec ParTranpsi(arma::vec psi,List Spec,int con_un,double chi){
    arma::vec psitrn=psi;
    //Gamma
    arma::uvec indGamma = Spec["indGamma"];
    psitrn.rows(indGamma-1)=ParTran(psi.rows(indGamma-1),-0.98*arma::ones<arma::vec>(indGamma.n_rows),0.98*arma::ones<arma::vec>(indGamma.n_rows),chi,con_un);
    //V
    arma::uvec indV = Spec["indV"];
    psitrn.rows(indV-1)=ParTran(psi.rows(indV-1),arma::zeros<arma::vec>(indV.n_rows),(arma::datum::inf)*arma::ones<arma::vec>(indV.n_rows),chi,con_un);
    //GQ
    arma::uvec indGQ = Spec["indGQ"];
    psitrn.rows(indGQ-1)=ParTran(psi.rows(indGQ-1),arma::zeros<arma::vec>(indGQ.n_rows),arma::ones<arma::vec>(indV.n_rows),chi,con_un);

    return(psitrn);
}


// [[Rcpp::export]]
arma::mat makeR0cpp(arma::mat G,arma:: mat Omega){
    int k=G.n_rows;
    int k2=k*k;
    arma::mat G2=arma::kron(G,G);
    arma::mat eyeG2=arma::eye(k2,k2)-G2;
    arma::vec megavec = arma::reshape(Omega,k2,1);
    arma::mat R00 = arma::inv(eyeG2)*megavec;
    arma::mat R0 = arma::reshape(R00,k,k);
    R0=(R0+arma::trans(R0))/2;
    return R0;
}





// [[Rcpp::export]]
double Kalman(arma::vec a, arma::mat b, arma::mat G,arma::mat Q, arma::mat Sigma,arma::mat ym){
    int T = ym.n_rows;
    int ntaum = ym.n_cols;
    int km = G.n_rows;
    arma::vec zerom=arma::zeros<arma::vec>(ntaum);
    arma::vec f_ll=arma::zeros<arma::vec>(km);
    arma::mat P_ll=makeR0cpp(G,Q);
    arma::mat ek= arma::eye<arma::mat>(km,km);
    arma::mat var_tl;
    arma::mat P_tl;
    arma::mat P_tt;
    arma::vec f_tl;
    arma::vec f_tt;
    arma::vec e_tl;
    arma::mat Kalgain;
    double lnL=0;

    for(int t=0;t<T;t++ ){

        f_tl = G*f_ll;
        P_tl = G*P_ll*arma::trans(G) + Q;
        var_tl = Sigma + b*P_tl*arma::trans(b);
        var_tl = 0.5*(var_tl + arma::trans(var_tl));
        e_tl = arma::trans(ym.row(t)) - a - b*f_tl;
        Kalgain = P_tl*arma::trans(b);
        arma::mat var_tlinv;
        if(arma::inv(var_tlinv,var_tl)==false){
            var_tlinv=invpd(var_tl);
        }
        Kalgain = Kalgain*var_tlinv;
        f_tt = f_tl + Kalgain*e_tl;
        P_tt = (ek - Kalgain*b)*P_tl;
        lnL = lnL + pdflogmvn(e_tl,zerom,var_tl);
        //Rcout<<pdflogmvn(e_tl,zerom,var_tl)<<endl;

        f_ll = f_tt;
        P_ll = P_tt;
    }

    return(lnL);
}


//[[Rcpp::export]]
arma::vec makeKcpp(arma::vec theta,List Spec){
    arma::uvec indK=Spec["indK"];
    int lmz = Spec["lmz"];
    arma::vec K=arma::zeros<arma::vec>(lmz);
    K.rows(0,lmz-1)=theta.rows(indK-1);
    return(K);
}

//[[Rcpp::export]]
arma::vec makeBetacpp(arma::vec theta, List Spec){
    int l =Spec["l"];
    arma::vec beta = arma::ones<arma::vec>(l);
    beta(l-1)=0;
    return(beta);


}

//[[Rcpp::export]]
arma:: mat makeGammacpp(arma::vec theta, List Spec)
{
    int lmz=Spec["lmz"];
    arma::uvec indGam=Spec["indGamma"];
    arma::mat Gamma = 1*arma::eye<arma::mat>(lmz,lmz)+0.5*arma::trimatu(arma::ones<arma::mat>(lmz,lmz));
    Gamma.elem(find(Gamma==0.5))=theta.rows(indGam-1);
    arma::mat Gamma1 = arma::trans(Gamma)+Gamma;
    Gamma1.diag(0)=arma::ones<arma::vec>(lmz);
    return(Gamma1);
}


//[[Rcpp::export]]
arma::mat makeOmegacpp(arma::mat V,arma::mat Gamma){
    arma::mat Omega=V*Gamma*V;
    return(Omega);
}

//[[Rcpp::export]]
arma::vec makeLambda0cpp(arma::vec theta, List Spec){

    arma::uvec indlambda0 =Spec["indlambda0"];
    arma::vec lambda0 = theta.rows(indlambda0-1);
    return(lambda0);
}

//G has to be transposed for posterior simulation
//[[Rcpp::export]]
arma::mat makeGcpp(arma::vec theta, List Spec){

   arma::uvec indG=Spec["indG"];
   int lmz=Spec["lmz"];
   arma::vec Gvec=theta.rows(indG-1);
   arma::mat G = arma::reshape(Gvec,lmz,lmz);
    return(arma::trans(G));

}

//[[Rcpp::export]]
arma::mat makeGQllcpp(arma::vec theta, arma::mat G, List Spec){
    int l = Spec["l"];
    arma::uvec indGQ = Spec["indGQ"];
    arma::vec kappa1=theta.rows(indGQ-1);
    double kappa = kappa1(0);
    arma::mat GQ = arma::zeros<arma::mat>(l,l);
    GQ.at(0,0)=1;
    GQ.at(1,1)= exp(-kappa);
    if(l==3){
        GQ.at(1,2)=kappa*(GQ.at(1,1));
        GQ.at(2,2)=GQ.at(1,1);
    }
    return(GQ);
}


//[[Rcpp::export]]
arma::mat makeGQllcpp2(arma::vec theta, arma::mat G, List Spec){
    int l = Spec["l"];
    arma::uvec indGQ = Spec["indGQ"];
    arma::mat GQ = 1*arma::eye<arma::mat>(l,l)+0.5*arma::trimatl(arma::ones<arma::mat>(l,l));
    arma::uvec indGQdiag=indGQ.rows(0,l-1);
    arma::uvec indGQoff=indGQ.rows(l,indGQ.n_rows-1);
    GQ.diag()=theta.rows(indGQdiag-1);
    GQ.elem(find(GQ==0.5))=theta.rows(indGQ-1);
    return(GQ);
}



//[[Rcpp::export]]
Rcpp::List makeABcpp(double delta, arma::mat GQll,arma::vec beta,arma::mat Ll,
                     arma::vec Lambda0,arma::uvec tau,arma::vec tau1,List Spec){

    int l = Spec["l"];
    int m = Spec["m"];
    int z = Spec["z"];
    int maxtau=max(tau);
    arma::mat B = arma::zeros<arma::mat>(l,maxtau);

    arma::vec BL=arma::zeros<arma::vec>(l,1);
    for(int indtau=0; indtau<maxtau; indtau++){
        B.col(indtau)=beta+arma::trans(GQll)*BL;
        BL=B.col(indtau);
    }

    arma::mat BOBm =arma::trans(B)*Ll*arma::trans(Ll)*B;
    arma::vec BOBm1=diagvec(BOBm);
    arma::vec BLLm=arma::trans(B)*Ll*(Lambda0);
    arma::vec dA = delta*arma::ones<arma::mat>(maxtau-1,1)-0.5*BOBm1.rows(0,maxtau-2)-BLLm.rows(0,maxtau-2);
    arma::vec dA1=arma::join_cols(delta*arma::ones<arma::mat>(1,1),dA);
    arma::vec A = cumsum(dA1);
    arma::vec a0=A.rows(tau-1)/tau1;
    arma::mat B1=arma::trans(B);
    arma::mat tmp=arma::kron(arma::ones<arma::rowvec>(l),tau1);
    arma::mat b0=B1.rows(tau-1)/tmp;
    arma::vec a = arma::join_cols(a0,arma::zeros(m+z,1));
    arma::mat b;
    arma::mat b1 = join_rows(b0,arma::zeros<arma::mat>(b0.n_rows,m+z));
    arma::mat temp1 = join_rows(arma::zeros(m+z,l),arma::eye<arma::mat>(m+z,m+z));
    b = arma::join_cols(b1,temp1);

    return(List::create(Rcpp::Named("a") = a,
                        Rcpp::Named("b") = b,
                        Rcpp::Named("A")=A,
                        Rcpp::Named("B")=B));
}







//[[Rcpp::export]]
double makeDeltacpp(arma::vec theta,List Spec){
    arma::uvec inddelta = Spec["inddelta"];
    arma::vec delta1= theta.rows(inddelta-1);
    double delta = delta1(0);
    return(delta);
}

//[[Rcpp::export]]
arma::mat makeGam_tilcpp(arma::mat Gam){
    arma::mat Gam_til = cholmod(Gam);
    return(Gam_til);
}



//[[Rcpp::export]]
arma::mat makeLlcpp(arma::mat Omega, List Spec){
    int l = Spec["l"];
    arma::mat Omegall;
    Omegall=Omega.submat(0,0,l-1,l-1);
    arma::mat Ll = cholmod(Omegall);
    return(Ll);
}



//[[Rcpp::export]]
arma:: vec makeThetacpp(arma::vec psi, List Spec){
    arma::vec theta = psi;
    double sf2= Spec["sf2"];
    arma::uvec indV=Spec["indV"];
    arma::uvec indSig=Spec["indSig"];
    theta.rows(indV-1)=0.0001*psi.rows(indV-1);
    theta.rows(indSig-1)=psi.rows(indSig-1)/sf2;
    return(theta);
}



//[[Rcpp::export]]
arma:: mat makeVcpp(arma::vec theta, List Spec){
    arma::uvec indV=Spec["indV"];
    arma::vec diagV = theta.rows(indV-1);
    arma::mat V = arma::diagmat(diagV);
    return(V);
}


//[[Rcpp::export]]
arma::mat makeSigmacpp(arma::vec theta,List Spec){
    int m = Spec["m"];
    int z = Spec["z"];
    arma::uvec indSig = Spec["indSig"];
    arma::vec diagSigma=theta.rows(indSig-1);
    arma::vec diagSigma1=arma::join_cols(diagSigma,arma::zeros<arma::vec>(m+z));
    arma::mat Sigma=arma::diagmat(diagSigma1);
    return(Sigma);
}

//[[Rcpp::export]]
double maker(double delta, arma::vec beta,arma::vec ft, List Spec){
    int l = Spec["l"];
    arma::vec lt = ft.rows(0,l-1);
    arma::vec temp = arma::trans(beta)*lt;
    double r = delta+temp(0);
    return(r);
}



//[[Rcpp::export]]
double maker2(int t, List Spec){
    //int l = Spec["l"];
    arma::mat ym = Spec["ym"];
    double r = ym(t,0);
    //arma::vec lt = ft.rows(0,l-1);
    //arma::vec temp = arma::trans(beta)*lt;
    //double r = delta+temp(0);
    return(r);
}






//Generate laten factor using basis yields method
//[[Rcpp::export]]
List Gen_Fmcpp(arma::vec psi,arma::mat Fm0,int ind_tsm,List Spec){
    arma::mat ym = Spec["ym"];
    arma::uvec tau = Spec["tau"];
    arma::vec tau1=Spec["tau"];
    arma::uvec basis=Spec["basis"];
    //arma::uvec indSig=Spec["indSig"];
    int lm = Spec["lm"];
    int l = Spec["l"];
    int z =Spec["z"];
    int lmz = Spec["lmz"];
    int m = lm-l;
    int T = ym.n_rows;
    int ntau = tau.n_rows;
    arma::mat Fm=arma::zeros<arma::mat>(T,lmz);
    //preparation
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    double delta = makeDeltacpp(theta, Spec);

    arma::mat G = makeGcpp(theta,Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::mat Ll=makeLlcpp(Q,Spec);
    arma::mat GQll = makeGQllcpp(theta, G, Spec);
    arma::vec beta  = makeBetacpp(theta, Spec);
    arma::mat Sigma = makeSigmacpp(theta, Spec);
    List AB=makeABcpp(delta,GQll,beta,Ll,Lambda0,tau,tau1,Spec);
    arma::vec a=as<arma::vec>(AB["a"]);
    arma::vec A=as<arma::vec>(AB["A"]);
    arma::mat b=as<arma::mat>(AB["b"]);
    arma::mat B=as<arma::mat>(AB["B"]);

    arma::vec f_tt;
    arma::mat P_tt;
    arma::vec ft=arma::zeros<arma::vec>(lmz);
    arma::vec a1=a.rows(basis-1);
    arma::vec ones= arma::ones<arma::vec>(T);
    arma::mat A1= a1*arma::trans(ones);
    arma::uvec temp=arma::zeros<arma::uvec>(l);
    temp(0)=0;
    temp(1)=1;
    temp(2)=2;
    arma::mat b1=b.submat(basis-1, temp);
    arma::mat y_basis=ym.cols(basis-1);
    arma::mat tempFm=arma::inv(b1)*(arma::trans(y_basis)-A1);
    Fm.cols(0,l-1)=arma::trans(tempFm);

    if(lmz>l){
    Fm.cols(l,lmz-1)=ym.cols(ntau,ntau+m+z-1);
    }

    //for (int j=0;j<T;j++){

        //arma::vec y= arma::trans(ym.row(j));
        //arma::vec y1=y.rows(basis-1);
        //arma::vec macro=y.rows(ntau,ntau+m+z-1);
        //ft.rows(0,l-1)=arma::solve(b1,y1-a1);
        //ft.rows(l,lmz-1)=macro;
        //Fm.row(j)=arma::trans(ft);

    //}


    if(arma::is_finite(Fm)<0.5){
        Fm=Fm0;
    }
    //make premium
    arma::mat Risk_tsm=arma::zeros<arma::mat>(Fm.n_rows,ntau-1);
    if(ind_tsm>0.5){
        for(int t=0;t<Fm.n_rows;t++){
            arma::vec fl=arma::trans(Fm.row(t));
            arma::vec ft=G*fl;
            arma::vec exr=arma::zeros<arma::vec>(max(tau)-1);
            //double rt=maker(delta,beta,fl,Spec);
            double rt=maker2(t,Spec);
            for(int indtau=1;indtau<max(tau);indtau++){
                arma::vec temp=arma::trans(B.col(indtau))*fl.rows(0,l-1)-arma::trans(B.col(indtau-1))*ft.rows(0,l-1);
                exr(indtau-1)=A(indtau)-A(indtau-1)+temp(0)-rt;
            }

            for(int indtau=1;indtau<tau.n_rows;indtau++){
                int ind = tau(indtau)-1;
                Risk_tsm(t,indtau-1)=sum(exr.rows(0,ind-1))/ind;
            }

        }
    }
    return (Rcpp::List::create(Rcpp::Named("Fm") = Fm,
                               Rcpp::Named("a") = a,
                               Rcpp::Named("b") = b,
                               Rcpp::Named("Risk_tsm")=Risk_tsm));

}


//[[Rcpp::export]]
List Gen_Gcpp(arma::vec psi,arma::mat Fm, List Spec){

    arma::uvec tau = Spec["tau"];
    arma::uvec indG=Spec["indG"];
    arma::vec tau1=Spec["tau"];
    arma::mat Gv_ = Spec["Gv_"];
    arma::mat G_=Spec["G_"];
    int T1=Fm.n_rows;
    arma::mat Fmc=Fm.rows(1,T1-1);
    arma::vec Fmcvec=arma::vectorise(Fmc);
    arma::mat Fml=Fm.rows(0,T1-2);
    int lmz = Spec["lmz"];
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::vec temp=arma::vectorise(Gv_);
    arma::vec G1_=arma::vectorise(G_);
    arma::mat G1v_=diagmat(temp);
    arma::mat XX=arma::trans(Fml)*Fml;
    arma::mat Gv=arma::inv(arma::inv(G1v_)+arma::kron(arma::inv(Q),XX));
    arma::vec Ghat=Gv*(arma::inv(G1v_)*G1_+arma::trans(arma::kron(arma::inv(Q),Fml))*Fmcvec);
    arma::vec Gvec=Ghat+cholmod(Gv)*rnorma(lmz*lmz);
    return (Rcpp::List::create(Rcpp::Named("Gvec") = Gvec,
                                   Rcpp::Named("Ghat") = Ghat,
                                   Rcpp::Named("Gv") = Gv));
}



//compute likelihood
//[[Rcpp::export]]
double lnL(arma::vec psi, List Spec){

    psi=ParTranpsi(psi,Spec,1,0.5);
    arma::mat ym = Spec["ym"];
    arma::uvec tau = Spec["tau"];
    arma::vec tau1=Spec["tau"];
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    double delta = makeDeltacpp(theta, Spec);
    arma::mat G = makeGcpp(theta,Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::mat Ll=makeLlcpp(Q,Spec);
    arma::mat GQll = makeGQllcpp(theta, G, Spec);
    arma::vec beta  = makeBetacpp(theta, Spec);
    arma::mat Sigma = makeSigmacpp(theta, Spec);
    List AB=makeABcpp(delta,GQll,beta,Ll,Lambda0,tau,tau1,Spec);
    arma::vec a=as<arma::vec>(AB["a"]);
    arma::mat b=as<arma::mat>(AB["b"]);
    double lnL=Kalman(a,b,G,Q,Sigma,ym);
    return(lnL);
}

//likelihood for mz
//[[Rcpp::export]]
double lnL3(arma::vec psi, List Spec){

    psi=ParTranpsi(psi,Spec,1,0.5);

    int l =Spec["l"];
    int m =Spec["m"];
    int z =Spec["z"];

    arma::mat ym = Spec["ym"];
    arma::uvec tau = Spec["tau"];
    arma::vec tau1=Spec["tau"];
    int ntau=tau.n_rows;
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    double delta = makeDeltacpp(theta, Spec);
    arma::mat G = makeGcpp(theta,Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::mat Ll=makeLlcpp(Q,Spec);
    arma::mat GQll = makeGQllcpp(theta, G, Spec);
    arma::vec beta  = makeBetacpp(theta, Spec);
    arma::mat Sigma = makeSigmacpp(theta, Spec);
    List AB=makeABcpp(delta,GQll,beta,Ll,Lambda0,tau,tau1,Spec);
    arma::vec a=as<arma::vec>(AB["a"]);
    arma::mat b=as<arma::mat>(AB["b"]);
    arma::mat ym_mz=ym.cols(l,l+m+z-1);
    arma::vec a1 = a.rows(ntau,ntau+m+z-1);
    arma::mat b1 = b.rows(ntau,ntau+m+z-1);
    arma::mat Sigma1=Sigma.submat(ntau,ntau,ntau+m+z-1,ntau+m+z-1);
    double lnL=Kalman(a1,b1,G,Q,Sigma1,ym_mz);
    return(lnL);
}



//yields only
//[[Rcpp::export]]
double lnL2(arma::vec psi, List Spec){

    psi=ParTranpsi(psi,Spec,1,0.5);
    arma::mat ym = Spec["ym"];
    arma::uvec tau = Spec["tau"];
    arma::vec tau1=Spec["tau"];
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    double delta = makeDeltacpp(theta, Spec);
    arma::mat G = makeGcpp(theta,Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::mat Ll=makeLlcpp(Q,Spec);
    arma::mat GQll = makeGQllcpp(theta, G, Spec);
    arma::vec beta  = makeBetacpp(theta, Spec);
    arma::mat Sigma = makeSigmacpp(theta, Spec);
    List AB=makeABcpp(delta,GQll,beta,Ll,Lambda0,tau,tau1,Spec);
    arma::vec a=as<arma::vec>(AB["a"]);
    arma::mat b=as<arma::mat>(AB["b"]);
    double lnL=Kalman2(a,b,G,Q,Sigma,ym);
    return(lnL);
}


//prior only for the MH_TARB part
//[[Rcpp::export]]
double lnprior(arma::vec psi,List Spec){
    psi=ParTranpsi(psi,Spec,1,0.5);
    arma::vec priorj=arma::zeros<arma::vec>(psi.n_rows);
    arma::uvec indGam = Spec["indGamma"];
    arma::uvec indV = Spec["indV"];
    arma::uvec indn = Spec["indn"];
    arma::vec mu = Spec["mu_"];
    arma::vec Var= Spec["Var_"];
    arma::vec nuV=Spec["nuV_"];
    arma::vec dV=Spec["dV_"];
    arma::vec gam = psi.rows(indGam-1);
    gam = (gam+arma::ones<arma::vec>(gam.n_rows))/2;
    for(int i=0; i<indn.n_elem;i++){
        priorj(indn(i)-1)= pdft1_(psi(indn(i)-1),mu(i),Var(i),15);
    }

    for(int i=0; i<indV.n_elem;i++){
        priorj(indV(i)-1)=pdfig1(psi(indV(i)-1),nuV(i)/2,dV(i)/2);
    }
    for(int i=0; i<indGam.n_elem; i++){
        priorj(indGam(i)-1)=lnpdfbeta(gam(i),20,20);
    }
    double sumpriorj = arma::sum(priorj);
    if(isnan(sumpriorj)==true){
        sumpriorj=-exp(20);
    }
   //sumpriorj=-exp(20);
    return (sumpriorj);
}

//[[Rcpp::export]]
double lnpriorfull(arma::vec psi,List Spec){
    psi=ParTranpsi(psi,Spec,1,0.5);
    arma::vec priorj=arma::zeros<arma::vec>(psi.n_rows);
    int z = Spec["z"];
    arma::uvec indGlz;

    if(z>0){

        indGlz=as<arma::uvec>(Spec["indGlz"]);

    }
    arma::uvec indGam = Spec["indGamma"];
    arma::uvec indV = Spec["indV"];
    arma::uvec indn = Spec["indn"];
    arma::vec mu = Spec["mu_"];
    arma::vec Var= Spec["Var_"];
    arma::vec nuV=Spec["nuV_"];
    arma::vec dV=Spec["dV_"];
    arma::uvec indG = Spec["indG"];
    arma::vec Gvec_=Spec["Gvec_"];
    arma::mat Gv_=Spec["Gv_"];
    arma::vec temp=arma::vectorise(Gv_);
    arma::mat G1v_=diagmat(temp);
    arma::vec gam = psi.rows(indGam-1);
    gam = (gam+arma::ones<arma::vec>(gam.n_rows))/2;
    double priorG=pdflogmvn(psi.rows(indG-1),Gvec_,G1v_);
    double priorGlz=0.0;
    if(z>0){
    priorGlz = pdflogmvn(psi.rows(indGlz-1),Gvec_.rows(indGlz-1),G1v_.submat(indGlz-1,indGlz-1));
    }
    priorj(0)=priorG-priorGlz;
    for(int i=0; i<indn.n_elem;i++){
        priorj(indn(i)-1)= pdft1_(psi(indn(i)-1),mu(i),Var(i),15);
    }
    for(int i=0; i<indV.n_elem;i++){
        priorj(indV(i)-1)=pdfig1(psi(indV(i)-1),nuV(i)/2,dV(i)/2);
    }
    for(int i=0; i<indGam.n_elem; i++){
        priorj(indGam(i)-1)=lnpdfbeta(gam(i),20,20);
    }


    arma::uvec indSig = Spec["indSig"];
    arma::vec v_ = Spec["nuSig_"];
    arma::vec d_ = Spec["dSig_"];
    priorj.rows(indSig-1)=pdfig1mv(psi.rows(indSig-1),v_/2,d_/2);
    double sumpriorj = arma::sum(priorj);
    return (sumpriorj);
}

//[[Rcpp::export]]
int paramconst(arma::vec psi,List Spec){
    arma::uvec indV=Spec["indV"];
    arma::uvec indGam = Spec["indGamma"];
    arma::uvec indGQ = Spec["indGQ"];
    arma::uvec indSig = Spec["indSig"];

    arma::uvec validm=arma::ones<arma::uvec>(30);
    validm(0)=arma::is_finite(psi);
    //arma::vec temp =psi.rows(indGam-1);
    //if(max(abs(temp))>15.0){
        //validm(5)=0;
    //}
    psi = ParTranpsi(psi,Spec,1,0.5);
    validm(1)=arma::is_finite(psi);
    if(min(validm)>0.5){
        arma::vec theta=makeThetacpp(psi, Spec);
        //arma::mat G = makeGcpp(theta,Spec);

        //arma::vec eigG = getEigenValuesd(G,psi);

        //if(max(eigG)>0.98){
            //validm(1)=0;
        //}

        arma::mat Gam = makeGammacpp(theta,Spec);
        arma::vec eigGam = getEigenValuesd(Gam,psi);
        if(min(eigGam)<=0.01){
            validm(2)=0;
        }

        //if(max(theta(indGQ-1))>=1){
        //validm(3)=0;
        //}
        //if(lnL(psi,Spec)<100){
            //validm(5)=0;
        //}
        //if(min(theta(indGQ-1))<=0){
        //validm(4)=0;
        //}

        //if(min(psi(indV-1))<=0){
        //validm(5)=0;
        //}

        if(max(abs(theta(indGam-1)))>=0.96){
        validm(7)=0;
        }
    }
    int valid=min(validm);

    return valid;

}


//[[Rcpp::export]]
int paramconstG(arma::vec psi,List Spec){

    arma::uvec validm=arma::ones<arma::uvec>(30);
    validm(0)=arma::is_finite(psi);
    if(min(validm)>0.5){
        arma::vec theta=makeThetacpp(psi, Spec);
        arma::mat G = makeGcpp(theta,Spec);

        arma::vec eigG = getEigenValuesd(G,psi);

        if(max(eigG)>0.99){
            validm(1)=0;
        }

    }
    int valid=min(validm);

    return valid;

}

//[[Rcpp::export]]
arma::mat getHessiannew(arma::vec theta, arma::uvec indbj, List Spec){
    int k = indbj.n_rows;
    double lnpost0 = lnL(theta,Spec)+lnprior(theta,Spec);
    arma::vec xarg = theta.rows(indbj-1);
    arma::vec h =arma::zeros<arma::vec>(k);
    for(int i=0;i<k;i++){
        double temp = abs(xarg(i));
        h(i)=pow(2.22e-16,0.3333)*(max(temp,1e-6));
    }
    arma::vec xargh = theta.rows(indbj-1)+h;
    h=xargh-theta.rows(indbj-1);
    arma::mat ee = arma::diagmat(h);
    arma::vec g = arma::zeros<arma::vec>(k);

    for(int i=0;i<k;i++){
        arma::vec thetaee = theta;
        thetaee.rows(indbj-1)=xarg+ee.col(i);
        g(i)=lnL(thetaee,Spec)+lnprior(thetaee,Spec);
    }

    arma::mat H = h*arma::trans(h);
    for (int i=0;i<k;i++){

        for(int j = i;j<k;j++){
            arma::vec thetaeeij = theta;
            arma::vec xargeeij = xarg+ee.col(i)+ee.col(j);
            thetaeeij.rows(indbj-1)=xargeeij;
            double lnposte = lnL(thetaeeij,Spec)+lnprior(thetaeeij,Spec);
            H(i,j)=(lnposte-g(i)-g(j)+lnpost0)/H(i,j);
            H(j,i)=H(i,j);
        }
        H = 0.5*(H+arma::trans(H));
    }
    return H;
}

//[[Rcpp::export]]
arma::vec Gradpnew1(arma::vec psi0,arma::uvec indbj,List Spec){
    arma::vec grdd1;
    arma::vec x0 = psi0.rows(indbj-1);
    double f = lnL(psi0,Spec)+lnprior(psi0,Spec);
    int k = x0.n_rows;
    arma::vec grdd = arma::zeros<arma::vec>(k);
    arma::vec ax0 = abs(x0);
    arma::vec dax0;
    if(arma::max(arma::abs(x0))<=0.000001){
        dax0 = x0/ax0;
        if(is_finite(dax0)==0){
            dax0=arma::ones<arma::vec>(k);
        }
    }else{
        dax0=arma::ones<arma::vec>(k);
    }
    arma::vec temp = arma::ones<arma::vec>(k);
    arma::mat ma = arma::join_cols(arma::trans(ax0),(1e-2)*arma::trans(temp));
    arma::vec dh = (1e-8)*arma::trans(arma::max(ma,0))%dax0;
    arma::vec psi1;

    for(int i=0;i<k;i++){
        psi1 = psi0;
        psi1.row(indbj(i)-1)=psi1.row(indbj(i)-1)+dh(i);
        grdd.row(i)=lnL(psi1,Spec)+lnprior(psi1,Spec);
    }
    arma::vec f0 = arma::ones<arma::vec>(k)*f;

    grdd1= (grdd-f0)/dh;

    return(grdd1);
}


//newton optimization algorithm
//[[Rcpp::export]]
arma::vec DO_CKR2(arma::vec psi0,arma::uvec indbj,int maxiter,List Spec){
    arma::vec psi1 = psi0;
    int valid =paramconst(psi1,Spec);
    if(valid==0){
        Rcout<<"DO_CKR2 starts in not satisfying constraints"<<endl;
    }
    double db = 1;
    int iter = 1;
    double s =1;
    arma::vec g;
    arma::mat H;
    arma::mat Hinv;
    arma::vec db1;
    arma::vec x00;
    double fcd0;
    double fcd1;
    while((db>1e-6)&(iter<=maxiter)&(s>=1e-6)){

        g=-Gradpnew1(psi1,indbj,Spec);
        H=-getHessiannew(psi1,indbj,Spec);
        if(arma::inv(Hinv,H)==false){
            Hinv=invpd(H);
        }
        db1=-Hinv*g;
        s=1;
        double s_hat = s;
        x00 = psi1;
        fcd0 = lnL(x00,Spec)+lnprior(x00,Spec);
        x00.rows(indbj-1)=x00.rows(indbj-1)+s*db1/2;
        if(paramconst(x00,Spec)==0){


            while((s>1e-8)&(paramconst(x00,Spec)==0)){
                x00=psi1;
                x00.rows(indbj-1)=x00.rows(indbj-1)+s*db1/2;
                if(s<1e-6){
                    x00=psi1;
                    s = 0;
                }
                s_hat=s;
                s=s/2;
            }

            fcd1= lnL(x00,Spec)+lnprior(x00,Spec);

            if(fcd1<fcd0){
                x00=psi1;
                s_hat=0;
            }


        }
        fcd1=fcd0-1;
        s=s_hat;
        while(fcd1<fcd0){
            x00=psi1;

            if(s>0){
                x00.rows(indbj-1)=x00.rows(indbj-1)+s*db1/2;
            }

            valid=paramconst(x00,Spec);

            if((s<1e-6)&&(valid==1)){
                s=0;

            }


            if(valid==0){
                fcd1=fcd0-1;
            }else{
                fcd1=lnprior(x00,Spec)+lnL(x00,Spec);
            }
            s_hat=s;
            s=s/2;

        }
        s=s_hat;
        //psi1(indbj-1)=psi1(indbj-1)+s*db1/2;
        psi1=x00;
        if((paramconst(psi1,Spec))==0){
            Rcout<<"DO_CKR2 results in not satisfying constraints"<<endl;
            Rcout<<psi1<<endl;
        }

        iter=iter+1;
        db = max(g);

    }
    int isfin = is_finite(psi1);
    arma::vec mu = isfin*psi1+(1-isfin)*psi0;



    return mu;

}



//[[Rcpp::export]]
arma::vec recserar(arma::vec x, double y0, double a){
    int n = x.n_rows;
    arma::vec y = arma::zeros<arma::vec>(n);
    y(0)=y0;
    for(int i=1;i<n;i++){
        y(i)=a*y(i-1)+x(i);
    }
    return(y);
}

//simulated anealing optimation algorithm
//[[Rcpp::export]]
arma::vec SA_CRK2(arma::vec psi0,arma::uvec indbj,List Spec,int n,double IT,
                  double a,double b,int IM,int mr,arma::vec SF,double eps,double cs){
    double lnpost0 = lnprior(psi0,Spec)+lnL(psi0,Spec);
    if(paramconst(psi0,Spec)==0){
        Rcout<<"SA starts with not satisfying constraints"<<endl;
        Rcout<<psi0<<endl;
    }
    int maxiter = 1;
    //storage for the global max and global maximamum
    double lnpostg = lnpost0;
    double alpha;
    arma::vec argg = psi0;
    arma::vec tau;
    arma::vec m;
    arma::vec thetap;
    arma::vec trialm = arma::zeros<arma::vec>(indbj.n_rows);
    if(n>1){
        tau = recserar(arma::zeros<arma::vec>(n),IT,a);//temperature
        m = recserar(b*arma::ones<arma::vec>(n),IM,1);//stage length
    }

    arma::mat rate = arma::zeros<arma::mat>(indbj.n_rows,n);
    arma::mat acceptm = arma::zeros<arma::mat>(indbj.n_rows,n);
    int reject = 0;
    int j =1;//index of stage
    while(j<=n){
        int mj = m(j-1);
        double tauj = tau(j-1);
        arma::vec trialm = arma::zeros<arma::vec>(indbj.n_rows);
        int i =1;//index of iteration
        int ind;

        while(i<=mj & reject<=mr){
            int valid = 0;
            int tic=0;
            while(valid==0){
                thetap = psi0;
                arma::vec currentj = thetap(indbj-1);
                arma::vec newparamj = currentj;
                arma::vec temp = randitg(newparamj.n_rows,1)-1;
                ind = temp(0);
                //Rcout<<indbj<<endl;
                //Rcout<<valid<<endl;
                newparamj(ind) = currentj(ind)+rndn_()/SF(ind);
                thetap(indbj-1)=newparamj;
                valid = paramconst(thetap,Spec);
                //tic=tic+1;
                //if(tic>2000){
                    //thetap=psi0;
                    //valid=1;
                //}
            }
            trialm(ind)=trialm(ind)+1;
            double lnpostp = lnL(thetap,Spec)+lnprior(thetap,Spec);
            if(lnpostp>lnpostg){
                lnpostg=lnpostp;
                argg = thetap;
            }
            double dlnpost = lnpostp-lnpost0;
            if(dlnpost>eps){
                alpha = 1;
            }
            else{
                alpha = exp(dlnpost/tauj);
            }
            double accept = rndu_();
            if(accept<alpha){
                psi0 = thetap;
                lnpost0 = lnpostp;
                acceptm(ind,j-1)=acceptm(ind,j-1)+1;
                reject = 0;
            }
            else{
                reject = reject+1;
            }
            i=i+1;
        }
        if(reject>mr){
            break;
        }
        rate(ind,j-1)=acceptm(ind,j-1)/trialm(ind);
        if(rate(ind,j-1)>0.7){
            SF(ind)=SF(ind)/(1+cs*(rate(ind,j-1)-0.7)/0.3);
        }
        else if(rate(ind,j-1)<0.2){

            SF(ind)=SF(ind)*(1+cs*((0.2-rate(ind,j-1))/0.2));
        }
        j=j+1;
    }



    if(maxiter>0){
        if(paramconst(argg,Spec)==0){
            Rcout<<"SA results in not satsifying constraints"<<endl;
        }
        argg=DO_CKR2(argg,indbj,maxiter,Spec);

    }

    return(argg);
}


//proposal distribution for TaRB
//[[Rcpp::export]]
Rcpp::List Gen_proposal_TaRB(arma::vec psi,arma::uvec indbj, int nu, List Spec, List Control){
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];
    arma::vec SFv = SF*arma::ones<arma::vec>(indbj.n_rows);
    arma::vec psimx = SA_CRK2(psi, indbj, Spec, n, IT, a, b, IM, mr, SFv, eps, cs);

        return Rcpp::List::create(Rcpp::Named("psimx") = psimx);


    //proposal distribution


}

//MH-algorithm
//[[Rcpp::export]]
Rcpp::List mhstep(arma::vec psi, arma::uvec indbj,int nu, List Spec, List Control){
    double lnlik0 = lnL(psi,Spec);
    double lnprior0 = lnprior(psi,Spec);
    double lnpost0 = lnlik0+lnprior0;
    int accept = 0;
    List results = Gen_proposal_TaRB(psi,indbj, nu, Spec, Control);
    arma::vec psimx = results["psimx"];
    arma::mat inV = -getHessiannew(psimx,indbj,Spec);
    arma::mat V;
    arma::vec psi1=psi;

    int valid1=1;
    int valid2=1;
    int valid;

    arma::vec theta=psimx.rows(indbj-1);
    arma::vec t_var = rmvta(nu,theta,V);

    psi1.rows(indbj-1)=t_var;
    valid2 = paramconst(psi1,Spec);
    double lnlik1 = lnL(psi1,Spec);
    valid=min(valid1,valid2);
    if((valid>0.5)){
        double lnprior1= lnprior(psi1,Spec);
        double lnpost1 = lnlik1+lnprior1;
        double q1 = pdft1mv(psi1.rows(indbj-1),psimx.rows(indbj-1),inV,nu);
        double q0 = pdft1mv(psi.rows(indbj-1),psimx.rows(indbj-1),inV,nu);
        double loga = min(0.0,lnpost1 + q0 - lnpost0-q1);

        if(log(rndu_())<loga){
            lnlik0=lnlik1;
            lnpost0=lnpost1;
            accept=1;
            return Rcpp::List::create(Rcpp::Named("psi") = psi1,
                                      Rcpp::Named("accept")=accept,
                                      Rcpp::Named("lnpost")=lnpost0);
        }
        else{
            return Rcpp::List::create(Rcpp::Named("psi") = psi,
                                      Rcpp::Named("accept")=accept,
                                      Rcpp::Named("lnpost")=lnpost0);
        }

    }
    else{

        return Rcpp::List::create(Rcpp::Named("psi") = psi,
                                  Rcpp::Named("accept")=accept,
                                  Rcpp::Named("lnpost")=lnpost0);


    }
}


//random block
//[[Rcpp::export]]
arma::uvec rndper(arma::uvec y){
    int n = y.n_elem;
    arma::uvec z = y;
    int k = n;
    while(k>1){

        int i= std::ceil(rndu_()*k);
        int zi = z(i-1);
        int zk = z(k-1);
        z(i-1) = zk;
        z(k-1) = zi;
        k = k-1;
    }
    arma::uvec retf = z;
    return(z);
}


//random block
//[[Rcpp::export]]
Rcpp::List randupp(int nmh, double tp){
    arma::uvec upps;
    arma::uvec nb;

    if(nmh>1){
        upps=arma::zeros<arma::uvec>(nmh);
        upps(nmh-1)=1;
        for(int itr = 0; itr<nmh;itr++){
            double u = rndu_();
            if(u>tp){
                upps(itr)=1;
            }
        }
    }
    else if(nmh==1){
        upps=arma::zeros<arma::uvec>(nmh);
        upps(0)=1;
    }
    int numhblck = sum(upps);
    arma::uvec upp = arma::zeros<arma::uvec>(numhblck);
    int itr = 0;
    int jj = 0;
    while(itr<nmh){
        if(upps(itr)==1){
            upp(jj) = itr+1;
            jj = jj + 1;
        }
        itr = itr + 1;
    }
    if(numhblck>1){
        arma::uvec temp = arma::zeros<arma::uvec>(numhblck);
        temp.rows(1,numhblck-1)=upp.rows(0,numhblck-2);
        nb=upp-temp;
    }
    else{
        nb = arma::zeros<arma::uvec>(1);
        nb(0)=nmh;
    }
    arma::uvec upp_=arma::cumsum(nb);
    arma::uvec low = arma::zeros<arma::uvec>(numhblck);
    if(numhblck>1){
        low.rows(1,numhblck-1)=upp_.rows(0,numhblck-2);
        low = low+1;
    }
    else if(numhblck==1){
        low(0)=1;
    }

    return Rcpp::List::create(Rcpp::Named("upp_") = upp_,
                              Rcpp::Named("low") = low,
                              Rcpp::Named("numhblck") = numhblck);

}

//[[Rcpp::export]]
Rcpp::List randupp_fixedblock(int nmh,int B, double tp){
    arma::uvec upps;
    arma::uvec nb;
    int numhblck;
    while(nb.is_empty()==true||min(nb)<5){
    if(nmh>1){
        upps=arma::zeros<arma::uvec>(nmh);
        upps(nmh-1)=1;
        arma::uvec ret  = as<arma::uvec>(multisample(nmh-1,B-1));
        (upps(ret-1)).fill(1);
    }
    numhblck = sum(upps);
    arma::uvec upp = arma::zeros<arma::uvec>(numhblck);
    int itr = 0;
    int jj = 0;
    while(itr<nmh){
        if(upps(itr)==1){
            upp(jj) = itr+1;
            jj = jj + 1;
        }
        itr = itr + 1;
    }
    if(numhblck>1){
        arma::uvec temp = arma::zeros<arma::uvec>(numhblck);
        temp.rows(1,numhblck-1)=upp.rows(0,numhblck-2);
        nb=upp-temp;
    }
    else{
        nb = arma::zeros<arma::uvec>(1);
        nb(0)=nmh;
    }
    }

    arma::uvec upp_=arma::cumsum(nb);
    arma::uvec low = arma::zeros<arma::uvec>(numhblck);
    low.rows(1,numhblck-1)=upp_.rows(0,numhblck-2);
    low = low+1;

    return Rcpp::List::create(Rcpp::Named("upp_") = upp_,
                              Rcpp::Named("low") = low,
                              Rcpp::Named("numhblck") = numhblck);

}



//[[Rcpp::export]]
Rcpp::List randblocks(arma::uvec indv, double tp){

    arma::uvec indv1 = rndper(indv);
    List randuppresults = randupp(indv.n_rows,tp);
    arma::uvec upp_=randuppresults["upp_"];
    arma::uvec low=randuppresults["low"];
    return Rcpp::List::create(Rcpp::Named("upp_") = upp_,
                              Rcpp::Named("low") = low,
                              Rcpp::Named("indv")=indv1);
}

//[[Rcpp::export]]
Rcpp::List randblock_fixed(arma::uvec indv, int B, double tp){

    arma::uvec indv1 = rndper(indv);
    List randuppresults = randupp_fixedblock(indv.n_rows,B,tp);
    arma::uvec upp_=randuppresults["upp_"];
    arma::uvec low=randuppresults["low"];
    return Rcpp::List::create(Rcpp::Named("upp_") = upp_,
                              Rcpp::Named("low") = low,
                              Rcpp::Named("indv")=indv1);

}

//[[Rcpp::export]]
Rcpp::List Gen_Theta_TaRB(arma::vec psi,List Spec,List Control, double tp, arma::vec count,int nu){
    arma::uvec indRB = Spec["indRB"];
    arma::uvec indSig = Spec["indSig"];
    int accept;
    double lnpost1;
    List results=randblocks(indRB,tp);
    arma::uvec indv0 = results["indv"];
    arma::uvec upp0 = results["upp_"];
    arma::uvec low0 = results["low"];
    int nbmh = low0.n_rows;
    for(int indthj = 0; indthj<nbmh; indthj++){
        arma::uvec indbj = indv0.rows(low0(indthj)-1,upp0(indthj)-1);
        List mhresults = mhstep(psi,indbj,nu,Spec,Control);
        accept= mhresults["accept"];
        arma::vec psi1 = mhresults["psi"];
        lnpost1=as<double>(mhresults["lnpost"]);
        psi=psi1;
        //counting acceptance for eah parameter
        count.rows(indbj-1)=count.rows(indbj-1)+accept*arma::ones<arma::vec>(indbj.n_rows);
    }
    return Rcpp::List::create(Rcpp::Named("psi0") = psi,
                              Rcpp::Named("accept") = accept,
                              Rcpp::Named("count")=count,
                              Rcpp::Named("lnpost")=lnpost1);
}

//[[Rcpp::export]]
Rcpp::List Gen_Theta_TaRB_reducedrun(arma::uvec indunfixedpara,arma::vec psi,List Spec,List Control, double tp,int nu){

    arma::vec psi1;
    List results=randblocks(indunfixedpara,tp);
    arma::uvec indv0 = results["indv"];
    arma::uvec upp0 = results["upp_"];
    arma::uvec low0 = results["low"];
    int nbmh = low0.n_rows;
    for(int indthj = 0; indthj<nbmh; indthj++){
        arma::uvec indbj = indv0.rows(low0(indthj)-1,upp0(indthj)-1);
        List mhresults = mhstep(psi,indbj,nu,Spec,Control);
        arma::vec psi1 = mhresults["psi"];
    }
    return Rcpp::List::create(Rcpp::Named("psi0") = psi);
}


//[[Rcpp::export]]
List Gen_Sigma(arma::vec psi, arma::mat Fm, arma::vec a, arma::mat b,List Spec){
    arma::mat ym = Spec["ym"];
    arma::uvec indSig = Spec["indSig"];
    arma::vec v_ = Spec["nuSig_"];
    arma::vec d_ = Spec["dSig_"];
    double sf2 = Spec["sf2"];
    arma::vec tau = Spec["tau"];
    arma::vec Sigma2=arma::zeros<arma::vec>(tau.n_elem);
    arma::vec d_post = arma::zeros<arma::vec>(tau.n_elem);
    double T = ym.n_rows;
    arma::vec v1= v_+T;
    double d;
    arma::vec e;
    for(int j=0;j<tau.n_elem;j++){
        e = ym.col(j)-a(j)*arma::ones<arma::vec>(T)-Fm*arma::trans(b.row(j));
        d = d_(j)+sum(e%e)*sf2;
        d_post(j)=d;
        Sigma2(j)=max(rigamma(v1(j)/2,d/2),0.5);
        //Sigma2(j)=rigamma(v1(j)/2,d/2);

    }
    arma::vec psit = psi;
    if(max(Sigma2)<15){
        psit.rows(indSig-1)=Sigma2;
    }
    if(paramconst(psi,Spec)>0.5){

        psi=psit;
    }
    return Rcpp::List::create(Rcpp::Named("psi")=psi,
                              Rcpp::Named("d_post")=d_post,
                              Rcpp::Named("v1")=v1);
}


//[[Rcpp::export]]
double pdfdenominator(arma::vec psi,arma:: vec tranpsim, arma::uvec block,arma::mat V0,List Control,List Spec){

    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];
    int nu = Spec["nu"];
    arma::vec SFv = SF*arma::ones<arma::vec>(block.n_rows);
    arma:: vec psimx =SA_CRK2(psi,block, Spec, n, IT, a, b, IM, mr, SFv, eps, cs);
    arma::vec thetahat = psimx.rows(block-1);
    arma::mat inV = -getHessiannew(psimx,block,Spec);
    arma::mat V;
    if(arma::inv(V,inV)==true){
        V0 = V;

    }else{
        V = V0;
    }
    arma::vec t_var;
    t_var = rmvta(nu,thetahat,V);
    arma::vec psistar = psi;
    psistar.rows(block-1) = tranpsim.rows(block-1);
    psi.rows(block-1)=t_var;
    double lnlikg = lnL(psi,Spec);
    double lnpriorg= lnprior(psi,Spec);
    double lnpostg = lnlikg+lnpriorg;
    double lnlikstar = lnL(psistar,Spec);
    double lnpriorstar= lnprior(psistar,Spec);
    double lnpoststar = lnlikstar+lnpriorstar;
    double q0 = pdft1mv(psistar.rows(block-1),psimx.rows(block-1),inV,nu);
    double q1 = pdft1mv(psi.rows(block-1),psimx.rows(block-1),inV,nu);
    double a1 = min((lnpostg+q0-lnpoststar-q1),0.0);
    double pdfdenominator=a1;
    return(pdfdenominator);
}


//[[Rcpp::export]]
double pdfnumerator(arma::vec psi,arma:: vec tranpsim, arma::uvec block,arma::mat V0,List Control,List Spec){
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];
    int nu = Spec["nu"];
    arma::vec SFv = SF*arma::ones<arma::vec>(block.n_rows);
    arma::vec psimx = SA_CRK2(psi, block, Spec, n, IT, a, b, IM, mr, SFv, eps, cs);
    arma::vec thetahat = psimx.rows(block-1);
    arma::mat inV = -getHessiannew(psimx,block,Spec);

    arma::mat V;
    //check inverse matrix if singular switch to previous V
    if(arma::inv(V,inV)==true){
        V0 = V;
    }else{
        V = V0;
    }

    double lnpriorg= lnprior(psi,Spec);
    double lnlikg = lnL(psi,Spec);
    double lnpostg = lnlikg+lnpriorg;
    arma::vec psistar = psi;
    psistar.rows(block-1) = tranpsim.rows(block-1);
    double lnpriorstar= lnprior(psistar,Spec);
    double lnlikstar = lnL(psistar,Spec);
    double lnpoststar = lnlikstar+lnpriorstar;
    double q0 = pdft1mv(psi.rows(block-1),psimx.rows(block-1),inV,nu);
    double q1 = pdft1mv(psistar.rows(block-1),psimx.rows(block-1),inV,nu);
    double a1 =min(0.0,(lnpoststar+q0-lnpostg-q1));
    double pdfnumerator=q1+a1;
    return(pdfnumerator);
}




//MCMC for sampling and marginal likelihood computation
//[[Rcpp::export]]
Rcpp::List MCMC_main(arma::vec psi0, List Spec, List Control, double tp, int n0, int n1,int J1,int J2,int B){

    //load the parameters
    int lmz = Spec["lmz"];
    arma::vec tau = Spec["tau"];
    arma::uvec tau1 = Spec["tau1"];
    int l = Spec["l"];
    int z = Spec["z"];
    int m = Spec["m"];
    int lm=Spec["lm"];
    arma::mat ym = Spec["ym"];
    int nu = Spec["nu"];
    int T1 = ym.n_rows;
    int ntau = tau.n_rows;
    int taumz = ym.n_cols;
    arma::uvec indSig=Spec["indSig"];
    arma::uvec indRB = Spec["indRB"];
    arma::uvec indG = Spec["indG"];
    arma::uvec indGamma = Spec["indGamma"];
    arma::mat Gv_=Spec["Gv_"];
    arma::mat G_=Spec["G_"];

    //load the restrictions
    arma::uvec indGlz;
    if(z>0){

    indGlz=as<arma::uvec>(Spec["indGlz"]);

    }
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];


    arma::vec count = arma::zeros<arma::vec>(psi0.n_rows);

    //calculating marginal likelihood
    //storage for mcmc
    arma::mat posterior=arma::zeros<arma::mat>(psi0.n_rows,n1);
    arma::vec lnpost = arma::zeros<arma::vec>(n0+n1);
    arma::cube post_Fm = arma::zeros<arma::cube>(T1,lmz,n1);
    arma::mat post_v1 = arma::zeros<arma::mat>(ntau,n1);
    arma::mat post_d1 = arma::zeros<arma::mat>(ntau,n1);
    arma::cube risk_tsm = arma::zeros<arma::cube>(T1,ntau-1,n1);
    arma::vec psim;
    arma::vec tranpsim;
    arma::mat temp= arma::zeros<arma::mat>(T1,l);
    arma::mat macrom;
    arma::mat risktsm1;
    arma::mat Fm0;
    if(m+z>0){
    macrom = ym.submat(0,ntau,T1-1,taumz-1);
    Fm0 = arma::join_rows(temp,macrom);
    }
    Fm0=temp;
    arma::mat Fm;
    arma::vec psi1;

    List fm0ret = Gen_Fmcpp(psi0,Fm0,0,Spec);
    Fm = as<arma::mat>(fm0ret["Fm"]);

    //parameter transformation
    arma::vec psitran0 = ParTranpsi(psi0,Spec,0,0.5);

    for(int i=0; i<n0+n1;i++){

        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psitran0,Spec,Control,tp,count,nu);
        arma::vec psitran1 = thetaresults["psi0"];
        arma::vec count1 = thetaresults["count"];
        lnpost.row(i)=as<double>(thetaresults["lnpost"]);
        //Sampling Factors
        int ind_tsm=0;
        arma::vec a;
        arma::mat b;
        psi1 = ParTranpsi(psitran1,Spec,1,0.5);

        if(is_finite(psi1)==1){
            psi0 = psi1;
            psitran0 = psitran1;
        }

        //Sampling G
        psi0=ParTranpsi(psitran0,Spec,1,0.5);
        List fmresults = Gen_Fmcpp(psi0,Fm,ind_tsm,Spec);
        a = as<arma::vec>(fmresults["a"]);
        b = as<arma::mat>(fmresults["b"]);
        Fm = as<arma::mat>(fmresults["Fm"]);
        //risktsm1=as<arma::mat>(fmresults["Risk_tsm"]);
        psi1= psi0;
        List resultG = Gen_Gcpp(psi1,Fm,Spec);
        arma::vec Gvec = resultG["Gvec"];
        psi1.rows(indG-1)=Gvec;

        if(paramconstG(psi1,Spec)==1){
            psi0=psi1;
        }

        //Sampling Sigma
        List Sigmaresults= Gen_Sigma(psi0,Fm,a,b,Spec);
        if(i>=n0){
            post_d1.col(i-n0)=as<arma::vec>(Sigmaresults["d_post"]);
            post_v1.col(i-n0)=as<arma::vec>(Sigmaresults["v1"]);

        }
        psi0 = as<arma::vec>(Sigmaresults["psi"]);
        count1(indSig-1)=count1(indSig-1)+1;

        psitran1=ParTranpsi(psi0,Spec,0,0.5);

        if(paramconst(psitran1,Spec)==1){
            psitran0=psitran1;
        }


        if(i>=n0){
            posterior.col(i-n0)=psi0;
            //risk_tsm.slice(i-n0)=risktsm1;

        }
        count=count1;
        //Rcout<<i<<endl;
    }

    Rcout<<"margin likeli begins"<<endl;
    // get the mean
    psim= row_means(posterior);
    tranpsim=ParTranpsi(psim,Spec,0,0.5);

    //computation of marginal likelihood of Sigma
    arma::mat pdfSigma = arma::zeros<arma::mat>(indSig.n_rows,n1);
    for(int i=0;i<n1;i++){

        arma::vec d = post_d1.col(i);
        arma::vec v = post_v1.col(i);
        pdfSigma.col(i)=pdfig1mv(tranpsim.rows(indSig-1),v/2,d/2);
    }

    arma::vec pdfsigm = arma::zeros<arma::vec>(indSig.n_rows);
    for(int i=0;i<indSig.n_rows;i++){

        pdfsigm(i)=pdfavg(arma::trans(pdfSigma.row(i)));

    }

    arma::vec pdfG=arma::zeros<arma::vec>(J1);


    //2th reduced run computation of G's marginal likelihood
    psitran0=tranpsim;
    for(int k=0;k<J1;k++){

        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psitran0,Spec,Control,tp,count,nu);
        arma::vec psitran1 = thetaresults["psi0"];
        //Sampling Factors
        arma::vec a;
        arma::mat b;
        arma::vec psi1 = ParTranpsi(psitran1,Spec,1,0.5);
        if((is_finite(psi1)==1)&(paramconst(psitran1,Spec)==1)){

            psi0 = psi1;
            psitran0 = psitran1;
        }

            psi0=ParTranpsi(psitran0,Spec,1,0.5);
            List fmresults = Gen_Fmcpp(psi0,Fm,0,Spec);

            a = as<arma::vec>(fmresults["a"]);
            b = as<arma::mat>(fmresults["b"]);
            Fm = as<arma::mat>(fmresults["Fm"]);
            risktsm1=as<arma::mat>(fm0ret["Risk_tsm"]);

        psi1= psi0;
        List resultG = Gen_Gcpp(psi1,Fm,Spec);
        arma::vec Gvec = resultG["Gvec"];
        arma::mat Gv = resultG["Gv"];
        arma::vec Ghat= resultG["Ghat"];
        psi1.rows(indG-1)=Gvec;
        double pdfG1 =pdflogmvn(tranpsim.rows(indG-1),Ghat,Gv);
        double pdfGlz=0.0;

        if(z!=0){
        pdfGlz = pdflogmvn(tranpsim.rows(indGlz-1),Ghat.rows(indGlz-1),Gv.submat(indGlz-1,indGlz-1));
        }

        pdfG.row(k)=pdfG1-pdfGlz;

        if(((paramconstG(psi1,Spec)==1))){

              psitran0.rows(indG-1)=Gvec;

          }

    }
    //3th reduced run marginal likelihood for indRB
    arma::uvec low =arma::zeros<arma::uvec>(B);
    arma::uvec upp =arma::zeros<arma::uvec>(B);

    List blockresults= randblock_fixed(indRB,B,0.2);
    low = as<arma::uvec>(blockresults["low"]);
    upp = as<arma::uvec>(blockresults["upp_"]);
    arma::uvec indvv = as<arma::uvec>(blockresults["indv"]);
    arma::uvec block1 = indvv.rows(low(0)-1,upp(0)-1);
    //creating storage
    arma::vec numerator = arma::zeros<arma::vec>(J2);
    arma::vec denominator = arma::zeros<arma::vec>(J2);
    arma::mat psi_reduced = arma::zeros<arma::mat>(psi0.n_rows,J2);
    arma::vec indRBpdf = arma::zeros<arma::vec>(B);

    arma::mat denominator1 = arma::zeros<arma::mat>(J2,B);
    arma::mat numerator1 = arma::zeros<arma::mat>(J2,B);


    arma::vec psi = tranpsim;
    //3th reduced run first block
    arma::mat V0 =0.01*arma::eye<arma::mat>(block1.n_rows,block1.n_rows);
    for(int j = 0; j<J2; j++){
        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psi,Spec,Control,tp,count,nu);
        arma::vec psi = as<arma::vec>(thetaresults["psi0"]);
        numerator1(j,0)=pdfnumerator(psi,tranpsim,block1,V0,Control,Spec);
        numerator(j)=numerator1(j,0);
    }
    //2th reduced run for the denominator
    arma::uvec block1f = indvv.rows(low(1)-1,upp(B-1)-1);
    psi = tranpsim;
   V0 =0.01*arma::eye<arma::mat>(block1.n_rows,block1.n_rows);
    for(int j = 0; j<J2; j++){
        //Sampling theta
        arma::vec psimx;
        arma::mat V;
        arma::mat inV;

        List thetaresults = Gen_Theta_TaRB_reducedrun(block1f,psi,Spec, Control,tp,nu);
        psi = as<arma::vec>(thetaresults["psi0"]);
        arma::vec temp=psi-tranpsim;
        psi_reduced.col(j)=psi;
        denominator1(j,0)=pdfdenominator(psi,tranpsim,block1,V0,Control,Spec);
        denominator(j)=denominator1(j,0);
    }
    indRBpdf(0) = pdfavg(numerator)-pdfavg(denominator);
    psi = tranpsim;
    //3th reduced run---B-1 reduced run

    for(int i= 1; i<B-1;i++){
        arma::uvec blockil = indvv.rows(low(0)-1,upp(i-1)-1);
        arma::uvec blocki = indvv.rows(low(i)-1,upp(i)-1);
        arma::vec numerator = arma::zeros<arma::vec>(J2);
        arma::vec denominator = arma::zeros<arma::vec>(J2);
        arma::uvec blockih = indvv.rows(low(i+1)-1,upp(B-1)-1);
        V0 =0.01*arma::eye<arma::mat>(blocki.n_rows,blocki.n_rows);

        //for the numerator
        for(int j = 0; j<J2; j++){
            psi = psi_reduced.col(j);
            numerator1(j,i)=pdfnumerator(psi,tranpsim,blocki,V0,Control,Spec);
            numerator(j)=numerator1(j,i);
        }
        //denominator
        psi = tranpsim;
        V0 =0.01*arma::eye<arma::mat>(blocki.n_rows,blocki.n_rows);

        for(int j = 0; j<J2; j++){

            List thetaresults = Gen_Theta_TaRB_reducedrun(blockih,psi,Spec, Control,tp,nu);
            psi = as<arma::vec>(thetaresults["psi0"]);
            psi_reduced.col(j)=psi;
            denominator1(j,i)=pdfdenominator(psi,tranpsim,blocki,V0,Control,Spec);
            denominator(j)=denominator1(j,i);

        }
        indRBpdf(i) = pdfavg(numerator)-pdfavg(denominator);
    }

    //for the last block B
    arma::uvec blockB = indvv.rows(low(B-1)-1,upp(B-1)-1);
    numerator = arma::zeros<arma::vec>(J2);
    denominator = arma::zeros<arma::vec>(J2);
    V0 =0.01*arma::eye<arma::mat>(blockB.n_rows,blockB.n_rows);
    for(int j = 0; j<J2; j++){
        psi = psi_reduced.col(j);
        numerator1(j,B-1)=pdfnumerator(psi,tranpsim,blockB,V0,Control,Spec);
        numerator(j)=numerator1(j,B-1);
    }
    //denominator
    psi = tranpsim;
    //Get the mode and Hessian
    V0 =0.01*arma::eye<arma::mat>(blockB.n_rows,blockB.n_rows);
    arma::vec SFv1 = SF*arma::ones<arma::vec>(blockB.n_rows);
    a=0.01;
    arma::vec psimx = SA_CRK2(psi,blockB, Spec, n, IT, a, b, IM, mr, SFv1, eps, cs);
    arma::vec thetahat = psimx.rows(blockB-1);
    arma::mat inV = -getHessiannew(psimx,blockB,Spec);
    arma::mat V;
    if(arma::inv(V,inV)==true){
        V0 = V;

    }else{

        V = V0;
        Rcout<<"V0"<<endl;
        Rcout<<V0<<endl;


    }
    for(int j = 0; j<J2; j++){

        psi = tranpsim;
        arma::vec t_var = rmvta(nu,thetahat,V);
        psi.rows(blockB-1)=t_var;
        arma::vec psistar=psi;
        psistar.rows(blockB-1) = tranpsim.rows(blockB-1);
        double lnlikg = lnL(psi,Spec);
        double lnlikstar = lnL(psistar,Spec);
        double lnpriorg= lnprior(psi,Spec);
        double lnpostg = lnlikg+lnpriorg;
        double lnpriorstar= lnprior(psistar,Spec);
        double lnpoststar = lnlikstar+lnpriorstar;
        double q0 = pdft1mv(psistar.rows(blockB-1),psimx.rows(blockB-1),inV,nu);
        double q1 = pdft1mv(psi.rows(blockB-1),psimx.rows(blockB-1),inV,nu);
        double a =min(0.0, lnpostg+q0-lnpoststar-q1);
        denominator(j)=a;
        denominator1(j,B-1)=denominator(j);

    }
    indRBpdf(B-1) = pdfavg(numerator)-pdfavg(denominator);
    double indRBpdfm = sum(indRBpdf);
    double likelim = lnL(tranpsim,Spec);
    double lnpriorm= lnpriorfull(tranpsim,Spec);
    double margilm = likelim+lnpriorm-sum(pdfsigm)-pdfavg(pdfG)-indRBpdfm;
    return Rcpp::List::create(Rcpp::Named("posterior") = posterior,
                              Rcpp::Named("pdfG")=pdfG,
                              Rcpp::Named("lnpost")= lnpost,
                              Rcpp::Named("count") = count,
                              Rcpp::Named("indRBpdf")=indRBpdf,
                              Rcpp::Named("pdfsigm")=pdfsigm,
                              Rcpp::Named("margilm")=margilm,
                              Rcpp::Named("pdfSigma")=pdfSigma,
                              Rcpp::Named("tranpsim")=tranpsim,
                              Rcpp::Named("risk_tsm")=risk_tsm,
                              Rcpp::Named("post_Fm")=post_Fm,
                              Rcpp::Named("Spec")=Spec,
                              Rcpp::Named("denominator1")=denominator1,
                              Rcpp::Named("numerator1")=numerator1,
                              Rcpp::Named("low")=low,
                              Rcpp::Named("upp")=upp,
                              Rcpp::Named("indvv")=indvv,
                              Rcpp::Named("indGlz")=indGlz);


}

//MCMC without computing marginal likelihood
//[[Rcpp::export]]
Rcpp::List MCMC_main1(arma::vec psi0, List Spec, List Control, double tp, int n0, int n1,int J1,int J2,int B){

    //load the parameters
    int lmz = Spec["lmz"];
    arma::vec tau = Spec["tau"];
    arma::uvec tau1 = Spec["tau1"];
    int l = Spec["l"];
    int z = Spec["z"];
    int m = Spec["m"];
    int lm=Spec["lm"];
    arma::mat ym = Spec["ym"];
    int nu = Spec["nu"];
    int T1 = ym.n_rows;
    int ntau = tau.n_rows;
    int taumz = ym.n_cols;
    arma::uvec indSig=Spec["indSig"];
    arma::uvec indRB = Spec["indRB"];
    arma::uvec indG = Spec["indG"];
    arma::uvec indGamma = Spec["indGamma"];
    arma::mat Gv_=Spec["Gv_"];
    arma::mat G_=Spec["G_"];
    arma::mat temp= arma::zeros<arma::mat>(T1,l);
    arma::mat macrom;
    arma::mat Fm0;
    //load the restrictions
    arma::uvec indGlz;
    if(z>0){

        indGlz=as<arma::uvec>(Spec["indGlz"]);

    }
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];
    if(m+z>0){
        macrom = ym.submat(0,ntau,T1-1,taumz-1);
        Fm0 = arma::join_rows(temp,macrom);
    }

    arma::vec count = arma::zeros<arma::vec>(psi0.n_rows);

    //calculating marginal likelihood
    //storage for mcmc
    arma::mat posterior=arma::zeros<arma::mat>(psi0.n_rows,n1);
    arma::vec lnpost = arma::zeros<arma::vec>(n0+n1);
    arma::cube post_Fm = arma::zeros<arma::cube>(T1,lmz,n1);
    arma::mat post_v1 = arma::zeros<arma::mat>(ntau,n1);
    arma::mat post_d1 = arma::zeros<arma::mat>(ntau,n1);
    arma::cube risk_tsm = arma::zeros<arma::cube>(T1,ntau-1,n1);
    arma::vec psim;
    arma::vec tranpsim;
    arma::mat Fm;
    arma::mat risktsm1;
    arma::vec psi1;

    List fm0ret = Gen_Fmcpp(psi0,Fm0,1,Spec);
    Fm = as<arma::mat>(fm0ret["Fm"]);
    //parameter transformation
    arma::vec psitran0 = ParTranpsi(psi0,Spec,0,0.5);

    for(int i=0; i<n0+n1;i++){

        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psitran0,Spec,Control,tp,count,nu);
        arma::vec psitran1 = thetaresults["psi0"];
        arma::vec count1 = thetaresults["count"];
        lnpost.row(i)=as<double>(thetaresults["lnpost"]);
        //Sampling Factors
        int ind_tsm=0;
        arma::vec a;
        arma::mat b;
        psi1 = ParTranpsi(psitran1,Spec,1,0.5);

        if(is_finite(psi1)==1){
            psi0 = psi1;
            psitran0 = psitran1;
        }

        //Sampling G
        psi0=ParTranpsi(psitran0,Spec,1,0.5);
        List fmresults = Gen_Fmcpp(psi0,Fm,ind_tsm,Spec);
        a = as<arma::vec>(fmresults["a"]);
        b = as<arma::mat>(fmresults["b"]);
        Fm = as<arma::mat>(fmresults["Fm"]);
        risktsm1=as<arma::mat>(fmresults["Risk_tsm"]);

        psi1= psi0;
        List resultG = Gen_Gcpp(psi1,Fm,Spec);
        arma::vec Gvec = resultG["Gvec"];
        psi1.rows(indG-1)=Gvec;

        if(paramconstG(psi1,Spec)==1){
            psi0=psi1;
        }

        //Sampling Sigma
        List Sigmaresults= Gen_Sigma(psi0,Fm,a,b,Spec);
        if(i>=n0){
            post_d1.col(i-n0)=as<arma::vec>(Sigmaresults["d_post"]);
            post_v1.col(i-n0)=as<arma::vec>(Sigmaresults["v1"]);
        }
        psi0 = as<arma::vec>(Sigmaresults["psi"]);
        count1(indSig-1)=count1(indSig-1)+1;

        psitran1=ParTranpsi(psi0,Spec,0,0.5);

        if(paramconst(psitran1,Spec)==1){
            psitran0=psitran1;
        }


        if(i>=n0){
            posterior.col(i-n0)=psi0;
            risk_tsm.slice(i-n0)=risktsm1;
        }
        count=count1;
    }

    psim= row_means(posterior);
    tranpsim=ParTranpsi(psim,Spec,0,0.5);

    return Rcpp::List::create(Rcpp::Named("posterior") = posterior,
                              Rcpp::Named("post_d1")=post_d1,
                              Rcpp::Named("post_v1")=post_v1,
                              Rcpp::Named("Control")=Control,
                              Rcpp::Named("lnpost")= lnpost,
                              Rcpp::Named("count") = count,
                              Rcpp::Named("tranpsim")=tranpsim,
                              Rcpp::Named("risk_tsm")=risk_tsm,
                              Rcpp::Named("post_Fm")=post_Fm,
                              Rcpp::Named("Spec")=Spec,
                              Rcpp::Named("indGlz")=indGlz,
                              Rcpp::Named("J1")=J1,
                              Rcpp::Named("J2")=J2);

}


//MCMC for marginal computation
//[[Rcpp::export]]
Rcpp::List MCMC_main2(arma::vec tranpsim, arma::mat posterior, List Spec, List Control, double tp,int J1,int J2,int B){

    //load the parameters
    int lmz = Spec["lmz"];
    arma::vec tau = Spec["tau"];
    arma::uvec tau1 = Spec["tau1"];
    int l = Spec["l"];
    int z = Spec["z"];
    int m = Spec["m"];
    int lm=Spec["lm"];
    arma::mat ym = Spec["ym"];
    int nu = Spec["nu"];
    int T1 = ym.n_rows;
    int ntau = tau.n_rows;
    int taumz = ym.n_cols;
    arma::uvec indSig=Spec["indSig"];
    arma::uvec indRB = Spec["indRB"];
    arma::uvec indG = Spec["indG"];
    arma::uvec indGamma = Spec["indGamma"];
    arma::mat Gv_=Spec["Gv_"];
    arma::mat G_=Spec["G_"];
    arma::vec count = arma::zeros<arma::vec>(tranpsim.n_rows);
    arma::mat macrom = ym.submat(0,ntau,T1-1,taumz-1);
    arma::mat temp= arma::zeros<arma::mat>(T1,l);
    arma::mat Fm0 = arma::join_rows(temp,macrom);
    arma::mat Fm;
    arma::vec psim= row_means(posterior);
    arma::vec psi0=psim;
    List fm0ret = Gen_Fmcpp(psi0,Fm0,0,Spec);
    Fm = as<arma::mat>(fm0ret["Fm"]);

    //load the restrictions
    arma::uvec indGlz;
    if(z>0){

        indGlz=as<arma::uvec>(Spec["indGlz"]);

    }
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];
    int n1 = posterior.n_cols;

    Rcout<<"margin likeli begins"<<endl;
    // get the mean
    //computation of marginal likelihood of Sigma
    arma::mat pdfSigma = arma::zeros<arma::mat>(indSig.n_rows,n1);
    //for(int i=0;i<n1;i++){

        //arma::vec d = post_d1.col(i);
        //arma::vec v = post_v1.col(i);
        //pdfSigma.col(i)=pdfig1mv(tranpsim.rows(indSig-1),v/2,d/2);
    //}

    arma::vec pdfsigm = arma::zeros<arma::vec>(indSig.n_rows);
    //for(int i=0;i<indSig.n_rows;i++){

        //pdfsigm(i)=pdfavg(arma::trans(pdfSigma.row(i)));

    //}

    arma::vec pdfG=arma::zeros<arma::vec>(J1);


    //2th reduced run computation of G's marginal likelihood
    arma::vec psitran0=tranpsim;
    for(int k=0;k<J1;k++){

        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psitran0,Spec,Control,tp,count,nu);
        arma::vec psitran1 = thetaresults["psi0"];
        //Sampling Factors
        arma::vec a;
        arma::mat b;
        arma::vec psi1 = ParTranpsi(psitran1,Spec,1,0.5);
        if((is_finite(psi1)==1)&(paramconst(psitran1,Spec)==1)){

            psi0 = psi1;
            psitran0 = psitran1;
        }

        psi0=ParTranpsi(psitran0,Spec,1,0.5);
        List fmresults = Gen_Fmcpp(psi0,Fm,0,Spec);

        a = as<arma::vec>(fmresults["a"]);
        b = as<arma::mat>(fmresults["b"]);
        Fm = as<arma::mat>(fmresults["Fm"]);

        psi1= psi0;
        List resultG = Gen_Gcpp(psi1,Fm,Spec);
        arma::vec Gvec = resultG["Gvec"];
        arma::mat Gv = resultG["Gv"];
        arma::vec Ghat= resultG["Ghat"];
        psi1.rows(indG-1)=Gvec;
        double pdfG1 =pdflogmvn(tranpsim.rows(indG-1),Ghat,Gv);
        double pdfGlz=0.0;

        if(z!=0){
            pdfGlz = pdflogmvn(tranpsim.rows(indGlz-1),Ghat.rows(indGlz-1),Gv.submat(indGlz-1,indGlz-1));
        }

        pdfG.row(k)=pdfG1-pdfGlz;

        if(((paramconstG(psi1,Spec)==1))){

            psitran0.rows(indG-1)=Gvec;

        }

    }
    //3th reduced run marginal likelihood for indRB
    arma::uvec low =arma::zeros<arma::uvec>(B);
    arma::uvec upp =arma::zeros<arma::uvec>(B);

    List blockresults= randblock_fixed(indRB,B,0.2);
    low = as<arma::uvec>(blockresults["low"]);
    upp = as<arma::uvec>(blockresults["upp_"]);
    arma::uvec indvv = as<arma::uvec>(blockresults["indv"]);
    arma::uvec block1 = indvv.rows(low(0)-1,upp(0)-1);
    //creating storage
    arma::vec numerator = arma::zeros<arma::vec>(J2);
    arma::vec denominator = arma::zeros<arma::vec>(J2);
    arma::mat psi_reduced = arma::zeros<arma::mat>(psi0.n_rows,J2);
    arma::vec indRBpdf = arma::zeros<arma::vec>(B);

    arma::mat denominator1 = arma::zeros<arma::mat>(J2,B);
    arma::mat numerator1 = arma::zeros<arma::mat>(J2,B);


    arma::vec psi = tranpsim;
    //3th reduced run first block
    arma::mat V0 =0.01*arma::eye<arma::mat>(block1.n_rows,block1.n_rows);
    for(int j = 0; j<J2; j++){
        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psi,Spec,Control,tp,count,nu);
        arma::vec psi = as<arma::vec>(thetaresults["psi0"]);
        numerator1(j,0)=pdfnumerator(psi,tranpsim,block1,V0,Control,Spec);
        numerator(j)=numerator1(j,0);
    }
    //2th reduced run for the denominator
    arma::uvec block1f = indvv.rows(low(1)-1,upp(B-1)-1);
    psi = tranpsim;
    V0 =0.01*arma::eye<arma::mat>(block1.n_rows,block1.n_rows);
    for(int j = 0; j<J2; j++){
        //Sampling theta
        arma::vec psimx;
        arma::mat V;
        arma::mat inV;

        List thetaresults = Gen_Theta_TaRB_reducedrun(block1f,psi,Spec, Control,tp,nu);
        psi = as<arma::vec>(thetaresults["psi0"]);
        arma::vec temp=psi-tranpsim;
        psi_reduced.col(j)=psi;
        denominator1(j,0)=pdfdenominator(psi,tranpsim,block1,V0,Control,Spec);
        denominator(j)=denominator1(j,0);
    }
    indRBpdf(0) = pdfavg(numerator)-pdfavg(denominator);
    psi = tranpsim;
    //3th reduced run---B-1 reduced run

    for(int i= 1; i<B-1;i++){
        arma::uvec blockil = indvv.rows(low(0)-1,upp(i-1)-1);
        arma::uvec blocki = indvv.rows(low(i)-1,upp(i)-1);
        arma::vec numerator = arma::zeros<arma::vec>(J2);
        arma::vec denominator = arma::zeros<arma::vec>(J2);
        arma::uvec blockih = indvv.rows(low(i+1)-1,upp(B-1)-1);
        V0 =0.01*arma::eye<arma::mat>(blocki.n_rows,blocki.n_rows);

        //for the numerator
        for(int j = 0; j<J2; j++){
            psi = psi_reduced.col(j);
            numerator1(j,i)=pdfnumerator(psi,tranpsim,blocki,V0,Control,Spec);
            numerator(j)=numerator1(j,i);
        }
        //denominator
        psi = tranpsim;
        V0 =0.01*arma::eye<arma::mat>(blocki.n_rows,blocki.n_rows);

        for(int j = 0; j<J2; j++){

            List thetaresults = Gen_Theta_TaRB_reducedrun(blockih,psi,Spec, Control,tp,nu);
            psi = as<arma::vec>(thetaresults["psi0"]);
            psi_reduced.col(j)=psi;
            denominator1(j,i)=pdfdenominator(psi,tranpsim,blocki,V0,Control,Spec);
            denominator(j)=denominator1(j,i);

        }
        indRBpdf(i) = pdfavg(numerator)-pdfavg(denominator);
    }

    //for the last block B
    arma::uvec blockB = indvv.rows(low(B-1)-1,upp(B-1)-1);
    numerator = arma::zeros<arma::vec>(J2);
    denominator = arma::zeros<arma::vec>(J2);
    V0 =0.01*arma::eye<arma::mat>(blockB.n_rows,blockB.n_rows);
    for(int j = 0; j<J2; j++){
        psi = psi_reduced.col(j);
        numerator1(j,B-1)=pdfnumerator(psi,tranpsim,blockB,V0,Control,Spec);
        numerator(j)=numerator1(j,B-1);
    }
    //denominator
    psi = tranpsim;
    //Get the mode and Hessian
    V0 =0.01*arma::eye<arma::mat>(blockB.n_rows,blockB.n_rows);
    arma::vec SFv1 = SF*arma::ones<arma::vec>(blockB.n_rows);
    a=0.01;
    arma::vec psimx = SA_CRK2(psi,blockB, Spec, n, IT, a, b, IM, mr, SFv1, eps, cs);
    arma::vec thetahat = psimx.rows(blockB-1);
    arma::mat inV = -getHessiannew(psimx,blockB,Spec);
    arma::mat V;
    if(arma::inv(V,inV)==true){
        V0 = V;

    }else{

        V = V0;
        Rcout<<"V0"<<endl;
        Rcout<<V0<<endl;


    }
    for(int j = 0; j<J2; j++){

        psi = tranpsim;
        arma::vec t_var = rmvta(nu,thetahat,V);
        psi.rows(blockB-1)=t_var;
        arma::vec psistar=psi;
        psistar.rows(blockB-1) = tranpsim.rows(blockB-1);
        double lnlikg = lnL(psi,Spec);
        double lnlikstar = lnL(psistar,Spec);
        double lnpriorg= lnprior(psi,Spec);
        double lnpostg = lnlikg+lnpriorg;
        double lnpriorstar= lnprior(psistar,Spec);
        double lnpoststar = lnlikstar+lnpriorstar;
        double q0 = pdft1mv(psistar.rows(blockB-1),psimx.rows(blockB-1),inV,nu);
        double q1 = pdft1mv(psi.rows(blockB-1),psimx.rows(blockB-1),inV,nu);
        double a =min(0.0, lnpostg+q0-lnpoststar-q1);
        denominator(j)=a;
        denominator1(j,B-1)=denominator(j);

    }
    indRBpdf(B-1) = pdfavg(numerator)-pdfavg(denominator);
    double indRBpdfm = sum(indRBpdf);
    double likelim = lnL(tranpsim,Spec);
    double lnpriorm= lnpriorfull(tranpsim,Spec);
    double margilm = likelim+lnpriorm-sum(pdfsigm)-pdfavg(pdfG)-indRBpdfm;
    return Rcpp::List::create(Rcpp::Named("posterior") = posterior,
                              Rcpp::Named("pdfG")=pdfG,
                              Rcpp::Named("count") = count,
                              Rcpp::Named("indRBpdf")=indRBpdf,
                              Rcpp::Named("pdfsigm")=pdfsigm,
                              Rcpp::Named("margilm")=margilm,
                              Rcpp::Named("pdfSigma")=pdfSigma,
                              Rcpp::Named("tranpsim")=tranpsim,
                              Rcpp::Named("Spec")=Spec,
                              Rcpp::Named("denominator1")=denominator1,
                              Rcpp::Named("numerator1")=numerator1,
                              Rcpp::Named("low")=low,
                              Rcpp::Named("upp")=upp,
                              Rcpp::Named("indvv")=indvv,
                              Rcpp::Named("indGlz")=indGlz);

}


//generarate yield prediction
//[[Rcpp::export]]
List yield_prediction(arma::mat Fm,arma::vec psi,arma::vec realized_yc,List Spec){

    arma::uvec tau = Spec["tau"];
    arma::vec tau1=Spec["tau"];
    arma::vec theta = makeThetacpp(psi, Spec);
    arma::vec Lambda0 = makeLambda0cpp(theta, Spec);
    double delta = makeDeltacpp(theta, Spec);

    arma::mat G = makeGcpp(theta,Spec);
    //arma::vec K=makeKcpp(theta,Spec);
    arma::mat V = makeVcpp(theta, Spec);
    arma::mat Gam = makeGammacpp(theta, Spec);
    arma::mat Q = makeOmegacpp(V, Gam);
    arma::mat Ll=makeLlcpp(Q,Spec);
    arma::mat GQll = makeGQllcpp(theta, G, Spec);
    arma::vec beta  = makeBetacpp(theta, Spec);
    arma::mat Sigma = makeSigmacpp(theta, Spec);
    List AB=makeABcpp(delta,GQll,beta,Ll,Lambda0,tau,tau1,Spec);
    arma::vec a=as<arma::vec>(AB["a"]);
    arma::vec A=as<arma::vec>(AB["A"]);
    arma::mat b=as<arma::mat>(AB["b"]);
    arma::mat B=as<arma::mat>(AB["B"]);

    //calculating the predictive likelihood
    int ntau = tau.n_rows;
    arma::vec factor_l=arma::trans(Fm.row(Fm.n_rows-1));
    arma::vec factor_f = G*factor_l;
    arma::vec mean_ycf = a+b*factor_f;
    arma::vec mean_yield= mean_ycf.rows(0,ntau-1);
    arma::mat b_y = b.rows(0,ntau-1);
    arma::mat var_ycf = b*Q*arma::trans(b)+Sigma;
    arma::mat var_ycf1 = (var_ycf+arma::trans(var_ycf))/2;
    //arma::mat L=cholmod(var_ycf1);

    arma::vec y=mean_ycf;
    arma::vec yield = y.rows(0,ntau-1);
    //double pred_density = pdflogmvn(realized_yc,mean_ycf,var_ycf1);
    double pred_density = pdflogmvn(realized_yc.rows(0,ntau-1),yield,var_ycf1.submat(0,0,ntau-1,ntau-1));
    //Rcout<<realized_yc.rows(0,ntau-1)<<endl;
    return Rcpp::List::create(Rcpp::Named("yield") = yield,
                              Rcpp::Named("pred_density")=pred_density);

}


//compute predictive likelihood
//[[Rcpp::export]]
List post_predictive(arma::vec psi0, arma::vec realized_yc, List Spec, List Control, double tp, int n0, int n1){

    //load the parameters
    int lmz = Spec["lmz"];
    arma::vec tau = Spec["tau"];
    arma::uvec tau1 = Spec["tau1"];
    int l = Spec["l"];
    int z = Spec["z"];
    int m = Spec["m"];
    int lm=Spec["lm"];
    arma::mat ym = Spec["ym"];
    int nu = Spec["nu"];
    int T1 = ym.n_rows;
    int ntau = tau.n_rows;
    int taumz = ym.n_cols;
    arma::uvec indSig=Spec["indSig"];
    arma::uvec indRB = Spec["indRB"];
    arma::uvec indG = Spec["indG"];
    arma::uvec indGamma = Spec["indGamma"];
    arma::mat Gv_=Spec["Gv_"];
    arma::mat G_=Spec["G_"];

    //load the restrictions
    arma::uvec indGlz;
    if(z>0){

        indGlz=as<arma::uvec>(Spec["indGlz"]);

    }
    int n =as<int>(Control["n"]);
    double IT = as<double>(Control["IT"]);
    double a =Control["a"];
    double b = Control["b"];
    int IM = Control["IM"];
    int mr = Control["mr"];
    double SF = Control["SF"];
    double eps =Control["eps"];
    int cs = Control["cs"];


    arma::vec count = arma::zeros<arma::vec>(psi0.n_rows);

    //calculating marginal likelihood
    //storage for mcmc
    arma::mat post_predictive1=arma::zeros<arma::mat>(n1,ntau);
    arma::vec lnPPLjm = arma::zeros<arma::vec>(n1);

    arma::vec psim;
    arma::vec tranpsim;
    arma::mat temp= arma::zeros<arma::mat>(T1,l);
    arma::mat macrom;
    arma::mat Fm0;
    if(m+z>0){
        macrom = ym.submat(0,ntau,T1-1,taumz-1);
        Fm0 = arma::join_rows(temp,macrom);
    }
    Fm0=temp;
    arma::mat Fm;
    arma::vec psi1;

    List fm0ret = Gen_Fmcpp(psi0,Fm0,0,Spec);
    Fm = as<arma::mat>(fm0ret["Fm"]);

    //parameter transformation
    arma::vec psitran0 = ParTranpsi(psi0,Spec,0,0.5);

    for(int i=0; i<n0+n1;i++){

        //Sampling theta
        List thetaresults = Gen_Theta_TaRB(psitran0,Spec,Control,tp,count,nu);
        arma::vec psitran1 = thetaresults["psi0"];
        arma::vec count1 = thetaresults["count"];
        //Sampling Factors
        int ind_tsm=0;
        arma::vec a;
        arma::mat b;
        psi1 = ParTranpsi(psitran1,Spec,1,0.5);

        if(is_finite(psi1)==1){
            psi0 = psi1;
            psitran0 = psitran1;
        }

        //Sampling G
        psi0=ParTranpsi(psitran0,Spec,1,0.5);
        List fmresults = Gen_Fmcpp(psi0,Fm,ind_tsm,Spec);
        a = as<arma::vec>(fmresults["a"]);
        b = as<arma::mat>(fmresults["b"]);
        Fm = as<arma::mat>(fmresults["Fm"]);

        psi1= psi0;
        List resultG = Gen_Gcpp(psi1,Fm,Spec);
        arma::vec Gvec = resultG["Gvec"];
        psi1.rows(indG-1)=Gvec;

        if(paramconstG(psi1,Spec)==1){
            psi0=psi1;
        }

        //Sampling Sigma
        List Sigmaresults= Gen_Sigma(psi0,Fm,a,b,Spec);
        psi0 = as<arma::vec>(Sigmaresults["psi"]);
        psitran1=ParTranpsi(psi0,Spec,0,0.5);

        if(paramconst(psitran1,Spec)==1){
            psitran0=psitran1;
        }

        if(i>=n0){
            List pred_results=yield_prediction(Fm,psi0,realized_yc,Spec);
            post_predictive1.row(i-n0)=arma::trans(as<arma::vec>(pred_results["yield"]));
            double pred_density=as<double>(pred_results["pred_density"]);
            lnPPLjm(i-n0)=pred_density;
            //Rcout<<pred_density<<endl;
        }

        count=count1;
    }

    //double lnPPLm =log(mean(exp(lnPPLjm)));
    double lnPPLm =pdfavg(lnPPLjm);

    return Rcpp::List::create(Rcpp::Named("post_predictive") = post_predictive1,
                              Rcpp::Named("lnPPLm")=lnPPLm);


}



