# -*- coding: utf-8 -*-


pip install d3rlpy

#@title
import numpy as np
from d3rlpy.dataset import MDPDataset
import d3rlpy

def state_generate(S_mean,A):
    S_mean_new = np.zeros(8)
    for i in range(4):
        S_mean_new[i] = (np.exp(A+S_mean[i])-np.exp(-A-S_mean[i]))/ (np.exp(A+S_mean[i])+np.exp(-A-S_mean[i]))
    for i in range(4,8):
        S_mean_new[i] = (np.exp(-A+S_mean[i])-np.exp(A-S_mean[i]))/ (np.exp(-A+S_mean[i])+np.exp(A-S_mean[i]))
    return S_mean_new
    

# def traj_generation(stage):
#     #np.random.seed(seed=seed)
#     A = np.random.uniform(low=0,high=1,size=(stage,1))
#     state = np.zeros((stage,2))
#     R = np.zeros(stage-1)
#     terminals = np.zeros(stage-1)
#     state[0] = np.random.normal(0, 0.5, 2)
#     for i in range(stage-1):
#         state[i+1] = state_generate1(A[i],state[i][0],state[i][1])
#         R[i] = -3*np.exp(state[1+i][0]-state[1+i][1])*(A[i]**2)+3*(state[1+i][0]+state[1+i][1]+0.5)*A[i]+3*state[1+i][0]+3*state[1+i][1]
#     state = state[0:-1]
#     A = A[0:-1]
#     terminals[-1] = 1
#     return state, A,R,terminals

# def data_generation(stage,n,seed):
#     np.random.seed(seed=seed)
#     res = traj_generation(stage)
#     state = res[0]
#     A = res[1]
#     R = res[2]
#     terminals =  res[3]
#     for i in range(n-1):
#         res = traj_generation(stage)
#         state = np.append(state,res[0],axis=0)
#         A = np.append(A, res[1],axis=0)
#         R = np.append(R, res[2])
#         terminals = np.append(terminals, res[3])
#     return state, A, R, terminals

def traj_test1(sac,stage_num,seed,sigma1):
    np.random.seed(seed=seed)
    s1 = np.random.normal(0, 0.5, 8)
    s_mean1=np.zeros(8)
    s_old = s1
    s_mean_old = s_mean1
    R = np.zeros(stage_num)
    disc = np.logspace(0, stage_num, num=stage_num,  endpoint=False, base=0.9, dtype=None)
    for i in range(stage_num):
        #action = sac.sample_action(s_old)
        action = sac.sample_action(s_old)
        s_mean_new = state_generate(s_mean_old, action)
        s_new = np.random.multivariate_normal(s_mean_new, sigma1, 1)[0]
        #print(s_new)
        R[i] = ((s_new[0]/2)**3+(s_new[1]/2)**3)+2*((s_new[4]/2)**3+(s_new[5]/2)**3)+s_new[2]+s_new[3]+0.5*(s_new[6]+s_new[7])
         #-3*np.exp(s_new[0]-s_new[1])*(action**2)+3*(s_new[0]+s_new[1]+0.5)*action+3*s_new[0]+3*s_new[1]
        s_old = s_new
        s_mean_old = s_mean_new
        #print(s_new)
    rew = sum(disc*R)
    return rew


import numpy as np
from d3rlpy.dataset import MDPDataset
import d3rlpy
# %load_ext rpy2.ipython


# %%R
# install.packages("mvtnorm")
# install.packages("randcorr")
# library(mvtnorm)
# library(randcorr)
# state_generate=function(s_mean,a){
# 	# s,s_mean 8-dims
# 	# a scaler
# 	s_mean_new=rep(0,8)
# 	for (i in 1:4){
# 		s_mean_new[i] = (exp(a+s_mean[i])-exp(-(a+s_mean[i])))/(exp(a+s_mean[i])+exp(-(a+s_mean[i])))
# 	}
# 	for (i in 5:8){
# 		s_mean_new[i] = (exp(-a+s_mean[i])-exp(-(-a+s_mean[i])))/(exp(-a+s_mean[i])+exp(-(-a+s_mean[i])))
# 	}
# 	return(s_mean_new)
# }
# 
# 
# data_generate=function(stage,n,dims=8,seed,sigma){
# 	set.seed(seed)
# 	A=runif(stage*n,min=-1,max=1)
# 	S=matrix(rep(0,stage*8*n),ncol=8)
# 	S_mean=matrix(rep(0,stage*8*n),ncol=8)
# 	S_mean[1:n,]=rnorm(8*n,mean=0,sd=0.5)
# 	for(i in 1:n){
# 		S[i,]=rmvnorm(1,mean=S_mean[i,],sigma=sigma)
# 	}
# 	for(i in (n+1):(n*stage)){
# 		S_mean[i,]=state_generate(S_mean[i-n,],A[i-n])
# 		S[i,]=rmvnorm(1,mean=S_mean[i,],sigma=sigma)
# 	}
# 	
# 	R=rep(0,n*(stage-1))
# 	for (i in 1:length(R)){
# 		R[i]=(S[n+i,1]/2)^3+(S[n+i,2]/2)^3+2*((S[n+i,5]/2)^3+(S[n+i,6]/2)^3)+S[n+i,3]+S[n+i,4]+0.5*(S[n+i,7]+S[n+i,8])
# 	}
# 
#   A_new=rep(0,(stage-1)*n)
#    R_new=rep(0,(stage-1)*n)
# 	 state_new=matrix(rep(0,n*8*(stage-1)),ncol=8)
# 
# 
#   for (i in 1:n){
#       ind = seq(from=i,by=n,length.out=stage-1)
#       state_new[c(((stage-1)*(i-1)+1):((stage-1)*i)),]=S[ind,]
#       A_new[c(((stage-1)*(i-1)+1):((stage-1)*i))]=A[ind]
#       R_new[c(((stage-1)*(i-1)+1):((stage-1)*i))]=R[ind]
#   }
# 
# 	terminals=rep(0,(stage-1)*n)
#   indd = seq(from=stage-1,by=stage-1,length.out=n)
#   terminals[indd]=1
# 	return(list(state=state_new,action=A_new,reward=R_new,terminals=terminals))
# 	
# 	
# }
# 
#

result=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R set.seed(seed)
#     %R sigma1=randcorr(8)/2
#     %R data=data_generate(stage=37,n=25,dims=8,seed=seed,sigma=sigma1)
#     %R seed=seed+1 
    data =%R data
    sigma1 =%R sigma1
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.DDPG(scaler="standard",action_scaler="min_max",batch_size=64, use_gpu=False,critic_learning_rate=0.0005)
    sac.fit(dataset,n_epochs=200,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] =traj_test1(sac,100,seed=j,sigma1=sigma1)
    result[i] = np.mean(result_tmp)
    print([i,result[i]])


result=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R set.seed(seed)
#     %R sigma1=randcorr(8)/2
#     %R data=data_generate(stage=37,n=25,dims=8,seed=seed,sigma=sigma1)
#     %R seed=seed+1 
    data =%R data
    sigma1 =%R sigma1
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.SAC(scaler="standard",action_scaler="min_max",batch_size=64, use_gpu=False,actor_learning_rate=0.001, critic_learning_rate=0.001)
    sac.fit(dataset,n_epochs=80,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] =traj_test1(sac,100,seed=j,sigma1=sigma1)
    result[i] = np.mean(result_tmp)
    print([i,result[i]])


result=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R set.seed(seed)
#     %R sigma1=randcorr(8)/2
#     %R data=data_generate(stage=25,n=25,dims=8,seed=seed,sigma=sigma1)
#     %R seed=seed+1 
    data =%R data
    sigma1 =%R sigma1
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.BEAR(scaler="standard",action_scaler="min_max",batch_size=64, use_gpu=False,actor_learning_rate=0.0003, critic_learning_rate=0.0003,temp_learning_rate=0.0003)
    sac.fit(dataset,n_epochs=70,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] =traj_test1(sac,100,seed=j,sigma1=sigma1)
    result[i] = np.mean(result_tmp)
    print([i,result[i]])
