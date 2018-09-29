ParzenK=function(z=z){

    K=0;
    if(z>=0 & z<=0.5){
        K=1-6*z^2+6*z^3;
    }
    if(z>0.5 & z<=1){

        K=2*(1-z)^3;
    }
    return(K);
}
