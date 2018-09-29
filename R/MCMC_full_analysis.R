##full MCMC analysis for Affine term structure models(including sampling of parameters and marginal likelihood computation

MCMC_full_analysis=function(relevant_factor=relevant_factor,
                           irrelevant_factor=irrelevant_factor,
                           dat=dat,
                           nu=nu,
                           n0=n0,
                           n1=n1,
                           J1=J1,
                           J2=J2,
                           tp=tp,
                           B=B){
    #load data
    ycm=dat[,1:8];
    #set up basis
    basis=c(1,5,8);
    #set up factors
    m=length(relevant_factor);
    z=length(irrelevant_factor);
    l=3;
    lmz=l+m+z;
    lm=l+m;
    m1=NULL;
    for(i in 1:m){
        m0=as.matrix(dat[relevant_factor[i]]);
        m0=m0-mean(m0);
        m1=cbind(m1,m0);
    }

    z1=NULL;
    for(i in 1:z){
        z0=as.matrix(dat[irrelevant_factor[i]]);
        z0=z0-mean(z0);
        z1=cbind(z1,z0);
    }

    macrom=cbind(m1,z1)
    ym=cbind(ycm,macrom);
    ym_full=ym;


    #yield curve
    tau=c(1,3,12,24,36,48,60,120);
    #tau=c(3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120);
    #prior
    s2m_=2;
    s2se_=0.5;
    nuSig_=reparaig(s2m_,s2se_)$a;
    dSig_=reparaig(s2m_,s2se_)$b;
    nuSig_=nuSig_*rep(1,length(tau));
    dSig_=dSig_*rep(1,length(tau));


    V_macro_=rep(2,m+z);
    V_=c(2,2.5,5,V_macro_);
    Vse_=0.25*rep(1,l+m+z);
    dV_=rep(0,length(V_));
    nuV_=rep(0,length(V_));
    for(i in 1:length(V_)){

        dV_[i]=reparaig(V_[i],Vse_[i])$b;
        nuV_[i]=reparaig(V_[i],Vse_[i])$a;

    }

    sf2=200000000;
    kappa_=-log(0.935);

    kappav_=0.001;

   #optimization setup
    IM=10;
    n=2;
    IT=5;
    a=0.01;
    b=10;
    mr = 400;
    cs = 20;
    eps = 1e-6;
    SF=50;
    Control1=list(n=n,
                  IT=IT,
                  a=a,
                  b=b,
                  SF=SF,
                  IM=IM,
                  mr=mr,
                  cs=cs,
                  eps=eps);


    #make the prior for G_ and Gv_
    G_=0.9*diag(lmz);
    Gvec_=matrix(G_,lmz*lmz,1);
    #Gv_=matrix(0.1,lmz,lmz);
    Gv_=matrix(0.1,lmz,lmz);
    if(z>0){
        Gv_[(l+m+1):(l+m+z),1:l]=0.000000001;
    }


    #prior for K
    #prior for delta
    delta_=mean(ym[,1]);
    deltav_=0.001;

    #prior for lambda
    lambda0_=c(-.17,-.07,0);
    lambda0v_=0.001*rep(1,l);

    #construct Spec
    nbG=lmz*lmz;
    nblambda0=l; #pricing risk
    nbGQ=1;
    nbd=1;
    nbV=lmz;
    nbGam=lmz*(lmz-1)/2;
    nbtau=length(tau);
    nb=c(nbG,nblambda0,nbGQ,nbd,nbV,nbGam,nbtau);
    nmh=sum(nb);
    indv=1:nmh;
    upp=cumsum(nb);
    low=rbind(0,matrix(upp[1:(length(nb)-1)],length(nb)-1,1))+1;



    indG=indv[low[1]:upp[1]];
    indlambda0=indv[low[2]:upp[2]];
    indGQ=indv[low[3]:upp[3]];
    inddelta=indv[low[4]:upp[4]];
    indV=indv[low[5]:upp[5]];
    indGamma=indv[low[6]:upp[6]];
    indSig=indv[low[7]:upp[7]];
    indRB=indv[low[2]:upp[6]];

    indn=c(indlambda0,indGQ,inddelta);
    mu_=c(lambda0_,kappa_,delta_);
    Var_=c(lambda0v_,kappav_,deltav_);

    psi_=rep(0,nmh);

    #create indicator for Glz
    indGlz=NULL;
    if(z==0){
        indGlz=NULL;
    }else{

    for(i in 1 : l){
        indGlz=c(indGlz, (lmz*(i-1)+lm+1):(lmz*(i-1)+lm+z));
        }
    }



    psi_[indG]=Gvec_;
    psi_[indlambda0]=lambda0_;
    psi_[indGQ]=kappa_;
    psi_[inddelta]=delta_;
    psi_[indV]=V_;
    psi_[indGamma]=0*rep(1,length(indGamma));
    psi_[indSig]=2*rep(1,length(indSig));
    Spec=list(
        tau=tau,
        tau1=tau,
        ym=as.matrix(ym),
        sf2=sf2,
        indG=indG,
        indlambda0=indlambda0,
        indGQ=indGQ,
        inddelta=inddelta,
        indV=indV,
        indGamma=indGamma,
        indSig=indSig,
        indRB = indRB,
        indn = indn,
        indGlz=indGlz,
        lmz=lmz,
        lm =lm,
        m=m,
        z=z,
        l=l,
        mu_=mu_,
        Var_=Var_,
        dV_=dV_,
        nuV_=nuV_,
        nuSig_=nuSig_,
        dSig_=dSig_,
        G_=G_,
        Gvec_=Gvec_,
        Gv_=Gv_,
        basis=basis,
        nu = nu,
        relevant_factor = relevant_factor,
        irrelevant_factor = irrelevant_factor);
    result= MCMC_main(psi_, Spec, Control1,tp,n0,n1,J1,J2,B);
    return(result);
}
