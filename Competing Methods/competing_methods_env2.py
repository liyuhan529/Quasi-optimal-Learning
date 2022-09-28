

pip install d3rlpy

import numpy as np
from d3rlpy.dataset import MDPDataset
import d3rlpy

def state_generate1(A,S1,S2):
  state1 = 0.75*(2*A-1)*S1+0.25*S1*S2+np.random.normal(0, 0.5, 1)
  state2 = 0.75*(1-2*A)*S2+0.25*S1*S2+np.random.normal(0, 0.5, 1)
  if (abs(state1)>2):
    state1 = np.sign(state1)*2
  if (abs(state2)>2):
    state2 = np.sign(state2)*2
  return np.concatenate((state1,state2))

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

def traj_test1(sac,stage_num,seed):
    np.random.seed(seed=seed)
    s1 = np.random.normal(0, 0.5, 2)
    s_old = s1
    R = np.zeros(stage_num)
    disc = np.logspace(0, stage_num, num=stage_num,  endpoint=False, base=0.9, dtype=None)
    for i in range(stage_num):
        #action = sac.sample_action(s_old)
        action = sac.predict(s_old)
        s_new = np.concatenate(state_generate1(action, s_old[0], s_old[1]))
                #print(action)
        R[i] = 0.25*(s_new[0]**3)+2*s_new[0]+0.5*(s_new[1]**3)+s_new[1]+0.25*(2*action-1)
         #-3*np.exp(s_new[0]-s_new[1])*(action**2)+3*(s_new[0]+s_new[1]+0.5)*action+3*s_new[0]+3*s_new[1]
        s_old = s_new
        #print(s_new)
    rew = sum(disc*R)
    return rew

import numpy as np
from d3rlpy.dataset import MDPDataset
import d3rlpy
# %load_ext rpy2.ipython


# %%R
# state_generate1=function(A,S1,S2){
# 	#	A_tmp=(A+100)/200
# 		state1=0.75*(2*A-1)*S1+0.25*S1*S2+rnorm(1,sd=0.5)
# 		state2=0.75*(1-2*A)*S2+0.25*S1*S2+rnorm(1,sd=0.5)
# 		if(abs(state1)>2){
# 			state1=sign(state1)*2
# 		}
# 		if(abs(state2)>2){
# 			state2=sign(state2)*2
# 		}
# 		return(c(state1,state2))
# 	}
# 	
# 
# 
# data_generation1=function(stage,n,dims=2,seed){
# 	set.seed(seed)
# 	A=runif(stage*n,max=1,min=0)
# 	#A_tmp=(A+100)/200
# 	state=matrix(rep(0,n*stage*dims),ncol=dims)
# 	
# 	state[1:n,]=rnorm(n*2, mean=0,sd=0.5)
# 	
# 	for(i in (n+1):(n*stage)){
# 		state[i,]=state_generate1(A[i-n],state[i-n,1],state[i-n,2])
# 	}
# 	
# 	R=rep(0,n*(stage-1))
# 	for (i in 1:length(R)){
# 		R[i]=0.25*state[n+i,1]^3+2*state[n+i,1]+0.5*state[n+i,2]^3+state[n+i,2]+0.25*(2*A[i]-1)
# 	}
# 
#   A_new=rep(0,(stage-1)*n)
#   R_new=rep(0,(stage-1)*n)
# 	state_new=matrix(rep(0,n*2*(stage-1)),ncol=2)
# 
# 
# 	 for (i in 1:n){
#       ind = seq(from=i,by=n,length.out=stage-1)
#       state_new[c(((stage-1)*(i-1)+1):((stage-1)*i)),]=state[ind,]
#       A_new[c(((stage-1)*(i-1)+1):((stage-1)*i))]=A[ind]
#       R_new[c(((stage-1)*(i-1)+1):((stage-1)*i))]=R[ind]
#   }
# 
# 	terminals=rep(0,(stage-1)*n)
#   indd = seq(from=stage-1,by=stage-1,length.out=n)
#   terminals[indd]=1
# 	return(list(state=state_new,action=A_new,reward=R_new,terminals=terminals))
# }
#


!pip install ipython-autotime
# %load_ext autotime


result=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R data=data_generation1(stage=37,n=50,seed=seed)
#     %R seed=seed+1
    data =%R data
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.BEAR(scaler="standard",action_scaler="min_max",batch_size=64, use_gpu=False,
                            actor_learning_rate=0.0001, critic_learning_rate=0.000001, imitator_learning_rate=0.00005, temp_learning_rate=0.0015, alpha_learning_rate=0.0001, 
                            actor_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False), 
                           critic_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False), 
                           temp_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False))
    sac.fit(dataset,n_epochs=80,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] = traj_test1(sac,100,seed=j)
    result[i] = np.mean(result_tmp)
    print([i,result[i]])


result=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R data=data_generation1(stage=37,n=50,seed=seed)
#     %R seed=seed+1
    data =%R data
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.SAC(scaler="standard",action_scaler="min_max",batch_size=64, use_gpu=False,
                            actor_learning_rate=0.0001, critic_learning_rate=0.000001, temp_learning_rate=0.0015,
                            actor_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False), 
                           critic_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False), 
                           temp_optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False))
    sac.fit(dataset,n_epochs=80,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] = traj_test1(sac,100,seed=j)
    result[i] = np.mean(result_tmp)
    print([i,result[i]])


result1=np.zeros(50)
# %R seed=1
for i in range(50):
#     %R data=data_generation1(stage=25,n=25,seed=seed)
#     %R seed=seed+1
    data =%R data
    #data = data_generation(25,25,1)
    observations = np.array(data[0])
    actions_tmp = np.array(data[1])
    actions = np.reshape(actions_tmp, (-1,1))
    rewards = np.array(data[2])
    terminals = np.array(data[3])

    dataset = MDPDataset(observations, actions, rewards, terminals)
    d3rlpy.seed(i)
    sac = d3rlpy.algos.DDPG(action_scaler="min_max",scaler="standard",batch_size=128, use_gpu=False,gamma=0.9,
                            actor_learning_rate=0.00001, critic_learning_rate=0.00001)
    sac.fit(dataset,n_epochs=50,save_metrics=False,show_progress=False,with_timestamp=False)
    result_tmp=np.zeros(100)
    for j in range(100):
        result_tmp[j] = traj_test1(sac,100,seed=j)
    result1[i] = np.mean(result_tmp)
    print([i,result1[i]])

