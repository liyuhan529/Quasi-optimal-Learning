# data generation for simulation setting 1
state_generate=function(A,S1,S2){
	#A_tmp=(A+100)/200
	state1=1*(2/(1+exp(-A))-1)*S1+0.25*S1*S2+rnorm(1,sd=0.5)
	state2=1*(1-2/(1+exp(-A)))*S2+0.25*S1*S2+rnorm(1,sd=0.5)
	#if(abs(state1)>2){
	#	state1=sign(state1)*2
	#}
	#if(abs(state2)>2){
	#	state2=sign(state2)*2
	#}
	return(c(state1,state2))
}

# quadratic form
data_generation=function(stage,n,dims=2,seed){
	set.seed(seed)
	A=runif(stage*n,max=1,min=0)
	#A_tmp=(A+100)/200
	state=matrix(rep(0,n*stage*dims),ncol=dims)
	
	state[1:n,]=rnorm(n*2, mean=0,sd=0.5)
	
	for(i in (n+1):(n*stage)){
		state[i,]=state_generate(A[i-n],state[i-n,1],state[i-n,2])
	}
	
	R=rep(0,n*(stage-1))
	for (i in 1:length(R)){
		R[i]=-3*exp(state[n+i,1]-state[n+i,2])*A[i]^2+3*(state[n+i,1]+state[n+i,2]+0.5)*A[i]+3*state[n+i,1]+3*state[n+i,2]
	}
	
	return(list(action=A,state=state,reward=R))
}


# code for testing

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
	# s is the state vector after normalization
	# a is the action
	# thetas are the estimated parameters
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
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1) # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	Q_a=function(a){alpha1*a^2+alpha2*a+alpha3}
	Q_int=integrate(Q_a,lower=lb,upper=ub)$value
	
	return(Q_int)
}

# learned policy given a state
pi_policy_tmp=function(s,a,lambda,theta1,theta2,theta3){
	alpha1=-exp(sum(theta1*basis_func_tmp(s)))
	alpha2=sum(theta2*basis_func_tmp(s))
	alpha3=sum(theta3*basis_func_tmp(s))
	tmp=-alpha2/(2*alpha1)
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1)  # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	K=ub-lb
	if (a>=lb & a<=ub){
		pi_a=Q_func_tmp(s=s,a=a,theta1=theta1,theta2=theta2,theta3=theta3)/(2*lambda)-Q_int_tmp(s=s,theta1=theta1,theta2=theta2,theta3=theta3,lambda=lambda,c1)/(2*lambda*K)+1/K
	}
	else{pi_a=0}
	return(pi_a)
}



traj=function(s1,theta1,theta2,theta3,stage_num,max_s,min_s,max_a,min_a,lambda){
	# s1 is the initial state 
	S=matrix(rep(0,length(s1)*stage_num),ncol=length(s1))
	S[1,]=s1
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
		# acceptance-rejection sampling
		a_samp=-1
		while(a_samp==-1){
			aa=runif(1,min=lb1,max=ub1)
			u=runif(1)
			tmp1=pi_policy_tmp(s=s_tmp,a=tmp,lambda=lambda,theta1=theta1,theta2=theta2,theta3=theta3)
			if (pi_policy_tmp(s=s_tmp,a=aa,lambda=lambda,theta1=theta1,theta2=theta2,theta3=theta3)/tmp1>u){
				a_samp=aa
			}
			else{a_samp=-1}
		}
		
		if(a_samp< 0){
			a_samp=0
		}else if (a_samp>1){
			a_samp=1}
		else{a_samp=a_samp}
		
		A[i] = a_samp*(max_a-min_a)+min_a
		#print(a_samp)
		#A_tmp = (A[i]+100)/200
		
		S[i+1,]=state_generate(A[i],S[i,1],S[i,2])
		#S[i+1,]=state_generate1(A[i],S[i,1],S[i,2])
		R[i]=-3*exp(S[1+i,1]-S[1+i,2])*A[i]^2+3*(S[1+i,1]+S[1+i,2]+0.5)*A[i]+3*S[1+i,1]+3*S[1+i,2]
		#R[i]=0.25*S[1+i,1]^3+2*S[1+i,1]+0.5*S[1+i,2]^3+S[1+i,2]+0.25*(2*A[i]-1)
	}
	return(list(S=S,A=A,R=R))
}



# repeat 50 times
simu_env1=function(simu_run,stage,N){
ss=rep(0,100)
r_rec=c()
learn_r=c()
for(i in 1:simu_run){
	#set.seed(i)
	dd = data_generation(stage=stage,n=N,seed=i)
	set.seed(3)
	params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,gamma=0.9,n=N,stage_num=stage,sigma=1,sigma_a=1,lambda=0.1,lr=0.002,tol=10^(-4),desc_rate = 0.0001,maxit=800)
	#disc=0.9^seq(from=0,to=98)
	#real_r[i]=mean(disc%*%aa)
	for (j in 1:100){
		set.seed(j)
		res=traj(rnorm(2,sd=0.5),theta1=params[1:19],theta2=params[20:38],theta3=params[39:57],stage_num=101,max_s=c(max(dd$state[,1]),max(dd$state[,2])), 
						 min_s = c(min(dd$state[,1]),min(dd$state[,2])),max_a=max(dd$action),min_a=min(dd$action),lambda=0.1)
		#r_rec=append(r_rec,res$R)
		disc=0.9^seq(from=0,to=99)
		ss[j]=sum(res$R*disc)
		
	}
	learn_r[i]=mean(ss)
	print(c(i,learn_r[i]))
}
return (learn_r)
}


