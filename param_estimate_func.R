#' @name param_estimate
#' @description The parameter estimation function for Entropy-regularized Reinforcement Learning in Continuous Action Space
#' @param S The state variable matrix
#' @param A A is the action vector corresponding to state matrix
#' @param R R is the reward vector corresponding to state matrix
#' @param lambda The tunning parameter controls the sparsity of induced policy
#' @param gamma The discounted factor
#' @param n The number of trajectories
#' @param stage_num The number of stages
#' @param sigma The kernel bandwith for state variables
#' @param sigma_a The kernel bandwith for action
#' @param nodes The nodes of B-spline
#' @param batch1 The batch size for SGD
#' @param epsilon The parameter to numerically approximate gradient
#' @param maxit The maximum iterations number for SGD
#' @param tol The stoppting criterion for SGD, measuring the L^2 distance of iterative theta
#' @param lr The learning rate for SGD
#' @param desc.rate The decreasing rate for the learning rate
#' @return a vector of learned parameters theta


param_estimate=function(S,A,R,lambda,gamma,n,stage_num,sigma=1,sigma_a=1,
												nodes=c(1:6/7),batch1=5,epsilon=10^(-6),maxit=500,tol=10^(-4),lr=0.002,desc_rate=0.0001){
	# nomalize data
	for (j in 1:dim(S)[2]){
		for (k in 1:dim(S)[1]){
			S[k,j] = (S[k,j]-min(S[,j]))/(max(S[,j])-min(S[,j]))
		}
	}
	A = (A-min(A))/(max(A)-min(A))
	
	basis = basis_func(S,nodes)
	
	 p=dim(S)[2]
	 loss_rec=rep(0,200)
	 theta_rec=matrix(nrow=200,ncol=28*p+3)
	 for(j in 1:200){
	 	theta_tmp=runif(28*p+3,min=0,max=1)
	 	theta_rec[j,]=theta_tmp
	 	loss_rec[j]=loss_func(theta=theta_tmp,S,A,R,lambda=lambda,gamma=gamma,basis=basis,n=n,stage_num=stage_num,sigma=sigma,sigma_a=sigma_a,
	 												nodes=c(1:6/7))
	 }
	 index=which.min(loss_rec)
	 theta_inits=theta_rec[index,]

	params=loss_func_bgd(theta=theta_inits,S,A,R,lambda=lambda,gamma=gamma,basis=basis,n=n,stage_num=stage_num,sigma=sigma,sigma_a=sigma_a,
											 nodes=c(1:6/7),batch1=batch1,epsilon=epsilon,maxit=maxit,tol=tol,lr=lr,desc_rate=desc_rate)
	return(params)
}

