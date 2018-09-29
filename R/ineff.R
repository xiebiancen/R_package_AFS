ineff=function(MHhat=MHhat,
               maxac=maxac){
    np=dim(MHhat)[2];
    acfm=matrix(0,nr=maxac,np);

    for(i in 1:np){
        acfm[,i]=acf1(x=MHhat[,i],maxac);
    }
    nh=dim(acfm)[1];
    ine=matrix(0,nr=np,nc=1);

    for(i in 1:np){
        sum_rho=rep(0,nh);

        for(j in 1:nh){

            sum_rho[j]=ParzenK(z=j/nh)*acfm[j,i];

        }
        ine[i]=1+2*sum(sum_rho);
    }

    return(ine);
}
