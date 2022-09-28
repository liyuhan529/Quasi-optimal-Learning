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
	# s is the state vector
	# a is the action
	# thetas are of the same length of basis
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
	#print(c(alpha1,alpha2,alpha3))
	tmp=-alpha2/(2*alpha1)
	#if((-alpha1/(12*lambda))^(1/3)*1.5<= 1/(2*c1*lambda)){
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1) # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	
	Q_a=function(a){alpha1*a^2+alpha2*a+alpha3}
	Q_int=integrate(Q_a,lower=lb,upper=ub)$value
	#}
	#else{
	#	fn <- function(x){
	#		c(x[2]^2*c1*(2*alpha1*x[2]^4/3+alpha1*x[1]^2*x[2]^2)-x[1]^2+c1*lambda,
	#			-x[2]^2*c1*(alpha1*x[2]^4/3+alpha1*x[1]^2*x[2]^2)-x[1]^2+c1*lambda-x[2]^2)
	#	}
	
	#	tmp_bd = (nleqslv(c(0.02,0.1), fn)$x)^2
	#	lb1 = tmp-sum(tmp_bd)
	#	ub1 = tmp+sum(tmp_bd)
	#	lb2 = tmp-tmp_bd[1]
	#	ub2 = tmp+tmp_bd[1]
	#	Q_a=function(a){alpha1*a^2+alpha2*a+alpha3}
	#	Q_int=2*integrate(Q_a,lower=lb1,upper=lb2)$value
	#}
	return(Q_int)
}

V_func_tmp=function(s,theta1,theta2,theta3,lambda,c1){
	alpha1=-exp(sum(theta1*basis_func_tmp(s)))
	alpha2=sum(theta2*basis_func_tmp(s))
	alpha3=sum(theta3*basis_func_tmp(s))
	tmp=-alpha2/(2*alpha1)
	#if((-alpha1/(12*lambda))^(1/3)*1.5<= 1/(2*c1*lambda)){
	lb=(-alpha2+(12*alpha1^2*lambda)^(1/3))/(2*alpha1) # add constraint
	ub=(-alpha2-(12*alpha1^2*lambda)^(1/3))/(2*alpha1)
	#print(lb)
	#print(ub)
	K=ub-lb
	
	tmp1 = (Q_int_tmp(s,theta1,theta2,theta3,lambda,c1)/(2*lambda*K)-1/K)^2*K
	tmp2 = alpha1^2*(ub^5-lb^5)/5+alpha1*alpha2*(ub^4-lb^4)/2+(alpha2^2+2*alpha1*alpha3)*(ub^3-lb^3)+alpha2*alpha3*(ub^2-lb^2)+alpha3^2*K
	result = lambda*(1-tmp1+tmp2/(4*lambda^2))
	return (result)
}


## sample code to reproduce real data analysis
patient1=read.csv("patient1.csv")

result=c()
obs=c()
des=rep(0.9^(0:47),27)
obs_rew=colSums(matrix(des*patient1$icg,nrow=48))
#set.seed(11)
for (i in 1:50){
	set.seed(i*100)
	uni = unique(patient1$day)
	ind = sample(uni,20)
	res=c()
	indd=patient1$day %in% ind
	train_new = patient1[indd,]
	train_new1 = train_new[order(train_new$tt),]
	X = as.matrix(train_new1[,c(4,5,7)])
	#A is the longitudinal assigned treatment 
	A = train_new1[,6]
	#R is the longitudinal reward 
	R = train_new1[-c((dim(X)[1]-19):dim(X)[1]),8]
	N=20
	fit1 = param_estimate(S=X,A=A,R=R,lambda=0.2,gamma=0.9,n=20,stage_num=48,sigma=0.5,sigma_a=0.5,
												nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=800,tol=10^(-5),lr=0.0005,desc_rate=0.0001)
	obs[i] = mean(obs_rew[which(uni%in%ind)])
	X_new=apply(X, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
	for(j in 1:N){
		res[j]=V_func_tmp(s=X_new[j,],theta1=fit1[1:28],theta2=fit1[29:56],theta3=fit1[57:84],lambda=0.75,c1=5)-2
	}
	result[i]=mean(res)
	print(c(i,obs[i],result[i]))
}
