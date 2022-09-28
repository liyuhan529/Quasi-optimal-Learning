#Stage Num
stage = 25
#Number of trajectories
N = 25


###  Environment I
data=data_generation(stage=stage,n=N,seed=1)

#S is the state variable matrix:
#eg. row1 : rowN is the state variables matrix at stage 1;
#eg. row(N+1) : row(2*N) is the state variables matrix at stage 2
#etc...
S = data$state

#A is the action vector
A = data$action
#R is the reward vector
R = data$reward

#Note that, S and A has 1 more stage than R, since at the end of stage, we only observe S and A but not R (as we don't know S^t+1). 


# parameter estimate
## output the value of objective function every 50 iterations
set.seed(1)
fit=param_estimate(S,A,R,gamma=0.9,n=N,stage_num=stage,sigma=1,sigma_a=1,lambda=0.1,lr=0.002,tol=3*10^(-4),desc_rate = 0.0001,maxit=800)


# generate one trajectory based on learned policy with length 100
set.seed(3)
res=traj(rnorm(2,sd=0.5),theta1=fit[1:19],theta2=fit[20:38],theta3=fit[39:57],stage_num=101,max_s=c(max(data$state[,1]),max(data$state[,2])), 
		 min_s = c(min(data$state[,1]),min(data$state[,2])),max_a=max(data$action),min_a=min(data$action),lambda=0.1)

# generate 100 trajectories based on learned policy and calculated the discounted sum of reward of each trajectory
res=c()
for(i in 1:100){
	set.seed(i)
	rr=traj(rnorm(2,sd=0.5),theta1=fit[1:19],theta2=fit[20:38],theta3=fit[39:57],stage_num=101,max_s=c(max(data$state[,1]),max(data$state[,2])), 
					min_s = c(min(data$state[,1]),min(data$state[,2])),max_a=max(data$action),min_a=min(data$action),lambda=0.1)
	des=0.9^(c(0:99))
	res[i]=sum(des*rr$R)
}
# mean cumulated reward of 100 trajectories
mean(res)

# repeat 50 times
ss=rep(0,100)
r_rec=c()
learn_r=c()
for(i in 1:50){
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







#### Environment II
#Stage Num
stage = 25
#Number of trajectories
N = 25

data=data_generation2(stage=stage,n=N,seed=1)

#S is the state variable matrix:
#eg. row1 : rowN is the state variables matrix at stage 1;
#eg. row(N+1) : row(2*N) is the state variables matrix at stage 2
#etc...
S = data$state

#A is the action vector
A = data$action
#R is the reward vector
R = data$reward

# parameter estimate
## output the value of objective function every 100 iterations
set.seed(11)
fit=param_estimate(S,A,R,gamma=0.9,n=N,stage_num=stage,sigma=1,sigma_a=1,lambda=0.05,lr=0.00005,tol=5*10^(-6),desc_rate = 0.0001,maxit=800)


# generate one trajectory based on learned policy with length 100
set.seed(8)
res=traj2(rnorm(2,sd=0.5),theta1=fit[1:19],theta2=fit[20:38],theta3=fit[39:57],stage_num=101,max_s=c(max(data$state[,1]),max(data$state[,2])), 
				 min_s = c(min(data$state[,1]),min(data$state[,2])),max_a=max(data$action),min_a=min(data$action),lambda=0.05)

# generate 100 trajectories based on learned policy and calculated the discounted sum of reward of each trajectory
res=c()
for(i in 1:100){
	set.seed(i)
	rr=traj2(rnorm(2,sd=0.5),theta1=fit[1:19],theta2=fit[20:38],theta3=fit[39:57],stage_num=101,max_s=c(max(data$state[,1]),max(data$state[,2])), 
					min_s = c(min(data$state[,1]),min(data$state[,2])),max_a=max(data$action),min_a=min(data$action),lambda=0.05)
	des=0.9^(c(0:99))
	res[i]=sum(des*rr$R)
}
# mean cumulated reward of 100 trajectories
mean(res)


# repeat 50 times
ss=rep(0,100)
learn_r=c()
for(i in 1:50){
	#set.seed(i)
	dd = data_generation2(stage=stage,n=N,seed=i)
	set.seed(11)
	params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,gamma=0.9,n=N,stage_num=stage,sigma=1,sigma_a=1,lambda=0.05,lr=0.00005,tol=5*10^(-6),desc_rate = 0.0001,maxit=800)
	#data=data_generate(stage=100,n=100,dim=8,seed=i,sigma=sigma1)
	#aa=matrix(data$reward, ncol=100,byrow=T)
	disc=0.9^seq(from=0,to=98)
	#real_r[i]=mean(disc%*%aa)
	for (j in 1:100){
		set.seed(j)
		res=traj2(rnorm(2,sd=0.5),theta1=params[1:19],theta2=params[20:38],theta3=params[39:57],stage_num=101,max_s=c(max(dd$state[,1]),max(dd$state[,2])), 
						 min_s = c(min(dd$state[,1]),min(dd$state[,2])),max_a=max(dd$action),min_a=min(dd$action),lambda=0.05)
		#r_rec=append(r_rec,res$R)
		disc=0.9^seq(from=0,to=99)
		ss[j]=sum(res$R*disc)
		
	}
	learn_r[i]=mean(ss)
	print(c(i,learn_r[i]))
}

# Environment III
#Stage Num
stage = 25
#Number of trajectories
N = 25


set.seed(4)
sigma1=randcorr(8)/2
dd=data_generate3(stage=stage,n=N,dims=8,2,sigma=sigma1)
params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,lambda=0.05,gamma=0.9,n=N,stage_num=stage,sigma=0.5,sigma_a=0.5,
											nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=500,tol=10^(-5),lr=0.00001,desc_rate=0.0001)

# generate 100 trajectories based on learned policy and calculated the discounted sum of reward of each trajectory
ss=rep(0,100)
for (j in 1:100){
	set.seed(j)
	res=traj3(state_dim=8,theta1=params[1:73],theta2=params[74:146],theta3=params[147:219],stage_num=101,
					 max_s=apply(dd$state,2,max),min_s=apply(dd$state,2,min),max_a=max(dd$action),min_a=min(dd$action),lambda=0.05,sigma=sigma1)
	#r_rec=append(r_rec,res$R)
	disc=0.9^seq(from=0,to=99)
	ss[j]=sum(res$R*disc)
}
mean(ss)

# repeat 50 times
ss=rep(0,100)
r_rec=c()
#real_r=c()
learn_r=c()
for(i in 1:50){
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


# Environment IV
#Stage Num
stage = 25
#Number of trajectories
N = 25

set.seed(2)
sigma1=randcorr(8)/2
dd=data_generate4(stage=stage,n=N,dims=8,2,sigma=sigma1)
set.seed(4)
params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,lambda=0.05,gamma=0.9,n=N,stage_num=stage,sigma=0.5,sigma_a=0.5,
											nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=500,tol=10^(-5),lr=0.00001,desc_rate=0.0001)

# generate 100 trajectories based on learned policy and calculated the discounted sum of reward of each trajectory
ss=rep(0,100)
for (j in 1:100){
	set.seed(j)
	res=traj4(state_dim=8,theta1=params[1:73],theta2=params[74:146],theta3=params[147:219],stage_num=101,
						max_s=apply(dd$state,2,max),min_s=apply(dd$state,2,min),max_a=max(dd$action),min_a=min(dd$action),lambda=0.05,sigma=sigma1)
	#r_rec=append(r_rec,res$R)
	disc=0.9^seq(from=0,to=99)
	ss[j]=sum(res$R*disc)
}
mean(ss)

# repeat 50 times
ss=rep(0,100)
r_rec=c()
#real_r=c()
learn_r=c()
for(i in 1:50){
	set.seed(i)
	sigma1=randcorr(8)/2
	dd = data_generate4(stage=stage,n=N,dim=8,seed=i,sigma=sigma1)
	set.seed(4)
	params=param_estimate(S=dd$state,A=dd$action,R=dd$reward,lambda=0.05,gamma=0.9,n=N,stage_num=stage,sigma=0.5,sigma_a=0.5,
												nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=800,tol=2*10^(-5),lr=0.00001,desc_rate=0.00001)
	for (j in 1:100){
		set.seed(j)
		res=traj4(state_dim=8,theta1=params[1:73],theta2=params[74:146],theta3=params[147:219],stage_num=101,
							max_s=apply(dd$state,2,max),min_s=apply(dd$state,2,min),max_a=max(dd$action),min_a=min(dd$action),lambda=0.05,sigma=sigma1)
		r_rec=append(r_rec,res$R)
		disc=0.9^seq(from=0,to=99)
		ss[j]=sum(res$R*disc)
		
	}
	learn_r[i]=mean(ss)
	print(c(i,learn_r[i]))
}




