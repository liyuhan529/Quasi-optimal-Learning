library(mvtnorm)
library(randcorr)
state_generate3=function(s_mean,a){
	# s,s_mean 8-dims
	# a scaler
	s_mean_new=rep(0,8)
	for (i in 1:4){
		s_mean_new[i] = (exp(a+s_mean[i])-exp(-(a+s_mean[i])))/(exp(a+s_mean[i])+exp(-(a+s_mean[i])))
	}
	for (i in 5:8){
		s_mean_new[i] = (exp(-a+s_mean[i])-exp(-(-a+s_mean[i])))/(exp(-a+s_mean[i])+exp(-(-a+s_mean[i])))
	}
	return(s_mean_new)
}



data_generate3=function(stage,n,dims=8,seed,sigma){
	set.seed(seed)
	A=runif(stage*n,min=-1,max=1)
	S=matrix(rep(0,stage*8*n),ncol=8)
	S_mean=matrix(rep(0,stage*8*n),ncol=8)
	S_mean[1:n,]=rnorm(8*n,mean=0,sd=0.5)
	for(i in 1:n){
		S[i,]=rmvnorm(1,mean=S_mean[i,],sigma=sigma)
	}
	for(i in (n+1):(n*stage)){
		S_mean[i,]=state_generate3(S_mean[i-n,],A[i-n])
		S[i,]=rmvnorm(1,mean=S_mean[i,],sigma=sigma)
	}
	
	R=rep(0,n*(stage-1))
	for (i in 1:length(R)){
		R[i]=-exp(S[n+i,1]/2+S[n+i,5]/2)*A[i]^2+2*(S[n+i,2]+S[n+i,3]+S[n+i,6]+S[n+i,7]+0.5)*A[i]+S[n+i,4]+S[n+i,8]
	}
	
	return(list(action=A,state=S,reward=R))
}

basis_func_tmp=function(s){
	# s is a row from state matrix after normalization
	basis=c(1)
	for (i in 1:length(s)){
		myknots=c(1:6/7)
		base=c(s[i],s[i]^2,s[i]^3,max(0,(s[i]-myknots[1]))^3,max(0,(s[i]-myknots[2]))^3,max(0,(s[i]-myknots[3]))^3,
					 max(0,(s[i]-myknots[4]))^3,max(0,(s[i]-myknots[5]))^3,max(0,(s[i]-myknots[6]))^3)
		basis=append(basis,base)
	}
	return(basis)
}

Q_func_tmp=function(s,a,theta1,theta2,theta3){

	alpha1=-exp(sum(theta1*basis_func_tmp(s)))
	alpha2=sum(theta2*basis_func_tmp(s))  ##any value
	alpha3=sum(theta3*basis_func_tmp(s))
	Q=alpha1*a^2+alpha2*a+alpha3
	return(Q)
}


Q_int_tmp=function(s,theta1,theta2,theta3,lambda,c1){
	alpha1=-exp(sum(theta1*basis_func_tmp(s)))
	alpha2=sum(theta2*basis_func_tmp(s))
	alpha3=sum(theta3*basis_func_tmp(s))
	tmp=-alpha2/(2*alpha1)
	#if((-alpha1/(12*lambda))^(1/3)*1.5<= 1/(2*c1*lambda)){
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1) # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	Q_a=function(a){alpha1*a^2+alpha2*a+alpha3}
	Q_int=integrate(Q_a,lower=lb,upper=ub)$value
	return(Q_int)
}


pi_policy_tmp=function(s,a,lambda,theta1,theta2,theta3,c1){
	alpha1=-exp(sum(theta1*basis_func_tmp(s)))
	alpha2=sum(theta2*basis_func_tmp(s))
	alpha3=sum(theta3*basis_func_tmp(s))
	tmp=-alpha2/(2*alpha1)
	#if((-alpha1/(12*lambda))^(1/3)*1.5<= 1/(2*c1*lambda)){
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1)  # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	K=ub-lb
	if (a>=lb & a<=ub){
		pi_a=Q_func_tmp(s=s,a=a,theta1=theta1,theta2=theta2,theta3=theta3)/(2*lambda)-Q_int_tmp(s=s,theta1=theta1,theta2=theta2,theta3=theta3,lambda=lambda,c1)/(2*lambda*K)+1/K
	}
	else{pi_a=0}

	return(pi_a)
}


traj3=function(state_dim,theta1,theta2,theta3,stage_num,max_s,min_s,max_a,min_a,lambda,sigma){
	# s1 is the initial state 
	S_mean=matrix(rep(0,state_dim*stage_num),ncol=state_dim)
	S=matrix(rep(0,state_dim*stage_num),ncol=state_dim)
	S_mean[1,]=rep(0,8)
	S[1,]=rmvnorm(1,mean=S_mean[1,],sigma=sigma)
	A=rep(0,length(stage_num))
	R=rep(0,length(stage_num)-1)
	for(i in 1:(stage_num-1)){
		s_old=S[i,]
		s_tmp=(s_old-min_s)/(max_s-min_s)
		alpha1=-exp(sum(theta1*basis_func_tmp(s_tmp)))
		alpha2=sum(theta2*basis_func_tmp(s_tmp))
		alpha3=sum(theta3*basis_func_tmp(s_tmp))
		tmp=-alpha2/(2*alpha1)
		
		
		lb1=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1)  # add constraint
		ub1=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
		

		# accept-reject sampling
		a_samp=-1
		while(a_samp==-1){
			aa=runif(1,min=lb1,max=ub1)
			u=runif(1)
			tmp1=pi_policy_tmp(s=s_tmp,a=tmp,lambda=lambda,theta1=theta1,theta2=theta2,theta3=theta3,c1=c1)
			if (pi_policy_tmp(s=s_tmp,a=aa,lambda=lambda,theta1=theta1,theta2=theta2,theta3=theta3,c1=c1)/tmp1>u){
				a_samp=aa
			}
			else{a_samp=-1}
		}
		if(a_samp< -1){
		a_samp=-1
		}else if (a_samp>2){
		 	a_samp=2}
		else{a_samp=a_samp}
		
		#a_samp=tmp
		A[i] = a_samp*(max_a-min_a)+min_a

		S_mean[i+1,]=state_generate3(S_mean[i,],A[i])
		S[i+1,] = rmvnorm(1,mean=S_mean[i,],sigma=sigma)
		
		R[i]=-exp(S[1+i,1]/2+S[1+i,5]/2)*A[i]^2+2*(S[1+i,2]+S[1+i,3]+S[1+i,6]+S[1+i,7]+0.5)*A[i]+S[1+i,4]+S[1+i,8]
	}
	return(list(S=S,A=A,R=R))
}


# repeat 50 times
simu_env3=function(simu_run,N,stage){
ss=rep(0,100)
r_rec=c()
#real_r=c()
learn_r=c()
for(i in 1:simu_run){
	set.seed(i)
	sigma1=randcorr(8)/2
	dd = data_generate3(stage=stage,n=N,dim=8,seed=i,sigma=sigma1)
	set.seed(4)
	params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,lambda=0.05,gamma=0.9,n=N,stage_num=stage,sigma=0.5,sigma_a=0.5,
												nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=800,tol=2*10^(-5),lr=0.00001,desc_rate=0.00001)
	for (j in 1:100){
		set.seed(j)
		res=traj3(state_dim=8,theta1=params[1:73],theta2=params[74:146],theta3=params[147:219],stage_num=101,
							max_s=apply(dd$state,2,max),min_s=apply(dd$state,2,min),max_a=max(dd$action),min_a=min(dd$action),lambda=0.05,sigma=sigma1)
		r_rec=append(r_rec,res$R)
		disc=0.9^seq(from=0,to=99)
		ss[j]=sum(res$R*disc)
		
	}
	learn_r[i]=mean(ss)
	print(c(i,learn_r[i]))
}
 return(learn_r)
}
