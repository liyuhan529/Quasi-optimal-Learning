
############---Environment Set-up---##############
library(RcppArmadillo)
library(Rcpp)
library(mvtnorm)
library(randcorr)


# Environment I
## stage should be 1 larger than assigned due to the data generation process
env1_n25_t24=simu_env1(simu_run=50,stage=25,N=25)
env1_n50_t24=simu_env1(simu_run=50,stage=25,N=50)
env1_n25_t36=simu_env1(simu_run=50,stage=37,N=25)
env1_n50_t36=simu_env1(simu_run=50,stage=37,N=50)

# Environment II
env2_n25_t24=simu_env2(simu_run=50,stage=25,N=25)
env2_n50_t24=simu_env2(simu_run=50,stage=25,N=50)
env2_n25_t36=simu_env2(simu_run=50,stage=37,N=25)
env2_n50_t36=simu_env2(simu_run=50,stage=37,N=50)


# Environment III
env3_n25_t24=simu_env3(simu_run=50,stage=25,N=25)
env3_n50_t24=simu_env3(simu_run=50,stage=25,N=50)
env3_n25_t36=simu_env3(simu_run=50,stage=37,N=25)
env3_n50_t36=simu_env3(simu_run=50,stage=37,N=50)

# Environment IV
env4_n25_t24=simu_env4(simu_run=50,stage=25,N=25)
env4_n50_t24=simu_env4(simu_run=50,stage=25,N=50)
env4_n25_t36=simu_env4(simu_run=50,stage=37,N=25)
env4_n50_t36=simu_env4(simu_run=50,stage=37,N=50)


