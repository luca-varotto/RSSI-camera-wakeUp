import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, PairwiseKernel)
from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.misc import derivative
from scipy.spatial import distance
from itertools import groupby
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings('ignore')

############################################################################################################################

from kalman_filter import Kalman_filter
from pf import PF

############################################################################################################################

# target visual POD wrt camera distance  
def p_D(d):
    d_s = 2
    beta_s = 5
    beta_l = 5
    d_l = 3.5
    d_opt = 3
    p_D = (  ( 1/(1+np.exp(beta_l*(d-d_l))) ) ) # ( 1/(1+np.exp(-beta_s*(d-d_s))) ) * 
    return p_D

# log-distance path-loss model 
def PLM(r0,n,d):
    return r0 - 10*n*np.log10(d)

# inverse log-distance path-loss model 
def PLM_inverse(r0,n,r):
    return 10**( (r0-r)/(10*n) ) 

# function to predict the POD, given the GP model and a RSSI value 
def f(x,model):
    return model.predict(np.array([x]).reshape(-1,1))[0][0]

# 
def integrand(x, model, rssi, sigma):
    return model.predict(np.array([x]).reshape(-1,1))[0][0] * scipy.stats.norm.pdf(x,rssi,sigma)

# compute classification accuracy as (TP+TN)/(TP+TN+FP+FN)
# indices[:,*] = [TP,FP,TN,FN]
def accuracy(indices): 
    MC_tests = np.shape(indices)[1]
    acc = -np.ones(MC_tests)
    for MCtest_idx in range(MC_tests):
        acc[MCtest_idx] = (indices[0,MCtest_idx] + indices[2,MCtest_idx])/ np.sum( np.array(indices[:,MCtest_idx]))
    # compute ecdf accuracy
    ecdf_acc = ECDF(acc)
    return acc, ecdf_acc

############################################################################################################################

                    # *** SETUP PARAMETERS ***
# target-camera distance
d_min = 0.5 # min target-camera distance
d_max = 5.5 # max target-camera distance

# synthetic RSSI generation according to the PLM
r0 = -35 # RSSI offset at d0 = 1 [m]
n = 2 # RSSI attenuation gain
sigma = 3 # RSSI noise std. dev.

fps = 10 # camera fps

N_train = int(9*1.0E+2) # train dataset cardinality
N_test = int(5*1.0E+2) # test duration
MC_tests = 50 # number of MC tests

# RSSI tracking via PF
N_s = 100 # number of particles
mu_omega = 0.0 # mean of the process model
sigma_omega = 0.1 # std. dev. of the process model 
pf_train = PF(N_s,\
            init_interval=[PLM(r0,n,d_max),PLM(r0,n,d_min)],\
            draw_particles_flag=False)

# Kalman filter to smooth POD measurements
x0 = 0.5 # initial guess
kf = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1) # A, B, C, Q, R, P

# devices specifications
T_RF = 0.1 # [s] Rx sampling time
P_rx = 20*1.0E-3 # [W] (V=3V, I = 7mA), Rx power consumption (see https://www.nordicsemi.com/-/media/Software-and-other-downloads/Product-Briefs/nRF52832-product-brief.pdf?la=en&hash=2F9D995F754BA2F2EA944A2C4351E682AB7CB0B9)
P_c_a = 100*1.0E-3 # [W], camera power consumption in active state (see "Energy Consumption Models for Smart Camera Networks", 
                # "Benefits of Wake-Up Radio in Energy-Efficient Multimodal Surveillance Wireless Sensor Network",
                # "Energy Characterization and Optimization of ImageSensing Toward Continuous Mobile Vision")
P_c_s = 1*1.0E-3 # [W], camera power consumption in sleep state (see "Energy Consumption Models for Smart Camera Networks", 
                # "Benefits of Wake-Up Radio in Energy-Efficient Multimodal Surveillance Wireless Sensor Network",
                # "Energy Characterization and Optimization of ImageSensing Toward Continuous Mobile Vision")
P_c_d = 100*1.0E-3 # [W], detection power consumption (see "Energy Characterization and Optimization of ImageSensing Toward Continuous Mobile Vision")
P_trans = 0.5*P_c_a # [W], state transition power consumption (see "Energy Consumption Models for Smart Camera Networks")
T_trans = 3*1.0E-3 # [s], transition time

                    # *** GP TRAINING ***
# train dataset generation
distance_train = np.empty((N_train,1))
pD_true = np.empty((N_train,1)) 
pD_meas = np.empty((N_train,1))
rssi_raw = np.empty((N_train,1))
rssi_smooth = np.empty((N_train,1))
d=d_max
for t in range(N_train):
    # true distance
    d = max(min(d+np.random.normal(-0.01,0.2),d_max),d_min)
    distance_train[t] = d
    # true POD (no RSSI noise, groundtruth knowledge on the POD function) --> groundtruth
    pD_true[t] = p_D(d)

    # visual detection event modeled as Bernoulli r.va. with parameter p_D(d), sampled fps times before a new RSSI comes
    # (hence, we have a Binomial experiment with parameters p_D(d) and fps)
    kf.predict()
    kf.update( np.random.binomial(fps, pD_true[t]) / fps ) # number of detections / number of frames (proportion of successes)
    pD_meas[t] = kf.x
    
    # RSSI predict
    pf_train.predict(mu_omega,sigma_omega)
    # RSSI sample
    rssi_raw[t] = PLM(r0,n,d) + np.random.normal(0,sigma)  
    # RSSI update
    pf_train.update(rssi_raw[t],sigma)
    # resampling
    pf_train.SIS_resampling()
    # compute MMSE estimate
    pf_train.estimation(est_type='MMSE')
    rssi_smooth[t] = pf_train.estimate
    
# plot 
fig = plt.figure(figsize=(9,6))
plt.subplot(4,1,1)
plt.plot(np.linspace(d_min,d_max,200), [p_D(d) for d in np.linspace(d_min,d_max,200)], \
    label=r'$p_D(d)$', linewidth=2)
plt.legend()
plt.xlabel(r"$d\;[m]$")
plt.subplot(4,1,2)
plt.plot(distance_train, label=r'$d_t$')
plt.xlabel(r"$t;[ms]$")
plt.ylabel(r"distance [m]")
plt.subplot(4,1,3)
plt.plot(rssi_raw,label=r"$\tilde{r}_t$") # raw RSSI
plt.plot(rssi_smooth,label=r"$\hat{r}_t^{PF}$") # smoothed RSSI
plt.legend()
plt.ylabel(r"RSSI $[dBm]$")
plt.xlabel(r"$t\;[ms]$")
plt.subplot(4,1,4)
plt.plot(pD_true,label=r'$p_{D,t}$') 
plt.plot(pD_meas,label=r'$\tilde{p}_{D,t}$') 
plt.xlabel(r"$t\;[ms]$")
plt.legend()
plt.tight_layout()
plt.show()

# define GP model
rssi_train = rssi_smooth
model_gp = GaussianProcessRegressor(Matern()+WhiteKernel(noise_level_bounds=(0.0,0.2)))
# fit the GP model
model_gp.fit(rssi_train,pD_meas)
idx = np.argsort(rssi_train,axis=0)
pD_est_gp,std_gp = model_gp.predict(rssi_train[idx,0],return_std=True)
pD_est_gp = np.clip(pD_est_gp,0.0,1.0)

# define NIGP model
f_der = np.zeros(N_train)
for i in range(N_train):
    f_der[i] = derivative(f,rssi_train[i],args=(model_gp,))
model_nigp = GaussianProcessRegressor(kernel=\
    Matern(length_scale=np.exp(model_gp.kernel_.theta)[0],length_scale_bounds="fixed") + \
    WhiteKernel(noise_level=np.exp(model_gp.kernel_.theta)[1],noise_level_bounds=(0.0,0.2))+\
    WhiteKernel(),alpha=f_der**2)
# fit the NIGP model
model_nigp.fit(rssi_train,pD_meas)
pD_est_nigp,std_nigp = model_nigp.predict(rssi_train[idx,0],return_std=True)
pD_est_nigp = np.clip(pD_est_nigp,0.0,1.0)

# plot training results
pD_ideal = [p_D(PLM_inverse(r0,n,r)) for r in rssi_train[idx,0]]
fig = plt.figure(figsize=(9,6))
# plt.subplot(2,1,1)
# plt.plot(rssi_train[idx,0], pD_ideal, label=r'$p_D(r)$',linewidth=2) # POD if RSSI were noiseless
# plt.plot(rssi_train[idx,0], pD_est_gp,c='g',label='$\widehat{p}_D(r)$',linewidth=2)
# plt.plot(rssi_train[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
#     alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
# plt.fill_between(np.squeeze(rssi_train[idx,0]),\
#         np.squeeze(pD_est_gp) - std_gp,\
#         np.squeeze(pD_est_gp) + std_gp,\
#         alpha=0.5, facecolor='g') 
# plt.legend(fontsize=30)
# plt.xlabel(r'$r\;[dBm]$',fontsize=35)
# # plt.fill_between(np.squeeze(rssi_train[idx,0]),\
# #         np.squeeze(pD_est_gp) - (std_gp + np.squeeze(f_der[idx]**2)),\
# #         np.squeeze(pD_est_gp) + (std_gp+ np.squeeze(f_der[idx]**2)),\
# #         alpha=0.2, facecolor='g', label='CI + noise')
print('LML: {:5.3f}'.format(model_gp.log_marginal_likelihood(model_gp.kernel_.theta)) + ', '\
    'R^2'+': {:4.3f}'.format(model_gp.score(rssi_train[idx,0],pD_ideal)) + ', '\
    'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_gp)))
    )
# plt.subplot(2,1,2)
plt.plot(rssi_train[idx,0], pD_ideal, label=r'$p_D(r)$',linewidth=2) # true RSSI-POD function
plt.plot(rssi_train[idx,0], pD_est_nigp,c='g',label='$\widehat{p}_D(r)$',linewidth=2) # estimated RSSI-POD function
plt.plot(rssi_train[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
    alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
plt.fill_between(np.squeeze(rssi_train[idx,0]),\
        np.squeeze(pD_est_nigp) - std_nigp,\
        np.squeeze(pD_est_nigp) + std_nigp,\
        alpha=0.5, facecolor='g') 
plt.legend(fontsize=30)
plt.xlabel(r'$r\;[dBm]$',fontsize=35)
print('LML: {:5.3f}'.format(model_nigp.log_marginal_likelihood(model_nigp.kernel_.theta)) + ', '\
    'R^2'+': {:4.3f}'.format(model_nigp.score(rssi_train[idx,0],pD_ideal)) + ', '\
    'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_nigp)))
        )
plt.xticks(np.arange(int(min(rssi_train)), int(max(rssi_train)), step=int((int(max(rssi_train))-int(min(rssi_train)))/5)),fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

                    # *** GP TESTING ***
# performance indices: TP, FP. TN, FN (for each MC test) 
indices_always = np.zeros((4,MC_tests)) # camera always active
indices_random = np.zeros((4,MC_tests)) # camera randomly active
indices_gp = np.zeros((4,MC_tests)) # proposed GP-based camera activation
indices_groundtruth = np.zeros((4,MC_tests)) # camera activation with groundtruth POD reconstrunction and no RSSI noise
# Hamming distance of camera state with groundtruth and other methods (for each MC test) 
ham_always = np.zeros(MC_tests)
ham_random = np.zeros(MC_tests)
ham_gp = np.zeros(MC_tests) 
# energy consumption
CE_always = np.zeros((N_test,MC_tests))
CE_random = np.zeros((N_test,MC_tests))
CE_gp = np.zeros((N_test,MC_tests))
CE_groundtruth = np.zeros((N_test,MC_tests))
E_tot = np.zeros((4,MC_tests))
# MC experiment
for MCtest_idx in tqdm(range(MC_tests)):

        # PF for RSSI tracking
        pf_test = PF(N_s,\
            init_interval=[PLM(r0,n,d_max),PLM(r0,n,d_min)],\
            draw_particles_flag=False)

        # single MC test
        rssi_raw_test = np.empty((N_test,1))
        rssi_smooth_test = np.empty((N_test,1))
        pD_true_test = np.empty((N_test,1))
        E_pD = np.empty((N_test,1))
        d = np.random.uniform(d_min,d_max) # initial target-camera distance
        camera_state = np.zeros((N_test,3),dtype=int) # random, GP, groundtruth
        for t in range(1,N_test):

            # RSSI sample
            rssi_raw_test[t-1] = PLM(r0,n,d) + np.random.normal(0,sigma)  
            # RSSI update
            pf_test.update(rssi_raw_test[t-1],sigma)
            # resampling
            pf_test.SIS_resampling()
            # compute MMSE estimate
            pf_test.estimation(est_type='MMSE')
            rssi_smooth_test[t-1] = pf_test.estimate

            d = max(min(d+np.random.normal(0.0,0.2),d_max),d_min)
            pD_true_test[t] = 1- (1-p_D(d))**fps
            pf_test.predict(mu_omega,sigma_omega)
            pD_predict = 1-(1- model_nigp.predict(pf_test.particles.reshape(-1,1),return_std=False))**fps
            E_pD[t] = np.average(pD_predict, weights=pf_test.weights, axis=0)

            # number of detections if the camera was always switched on 
            nb_detections = np.random.binomial(fps, pD_true_test[t])

            # indices for camera always active
            indices_always[0,MCtest_idx] += nb_detections  
            indices_always[1,MCtest_idx] += fps-nb_detections
            # "wrong" energy consumption for camera always active
            E = (P_c_a +P_c_d)*T_RF
            E_tot[0,MCtest_idx] += E
            CE_always[t,MCtest_idx] = CE_always[t-1,MCtest_idx] + E*((fps-nb_detections)/fps)

            # camera randomly active 
            if np.random.binomial(1, 0.5): # activate according to a coin toss 
                indices_random[0,MCtest_idx] += nb_detections
                indices_random[1,MCtest_idx] += fps-nb_detections
                E = (P_c_a +P_c_d)*T_RF
                CE_random[t,MCtest_idx] = CE_random[t-1,MCtest_idx] + E*((fps-nb_detections)/fps) 
                E_trans = 0 if camera_state[t-1,0] ==1 else P_trans*T_trans
                camera_state[t,0]=1
            else:
                indices_random[2,MCtest_idx] += fps-nb_detections
                indices_random[3,MCtest_idx] += nb_detections
                E = P_c_s*T_RF
                CE_random[t,MCtest_idx] = CE_random[t-1,MCtest_idx] +  E*((nb_detections)/fps)
                E_trans = 0 if camera_state[t-1,0] ==0 else P_trans*T_trans
                camera_state[t,0]=0
            # "wrong" energy consumption for camera randomly active
            CE_random[t,MCtest_idx] += E_trans
            E_tot[1,MCtest_idx] += (E+E_trans)
            
            E_max = (P_c_a+P_c_d+ P_rx)*T_RF + P_trans*T_trans # max energy consumption
            E_norm_a = (P_c_a+P_c_d+ P_rx)*T_RF / E_max if camera_state[t-1,1] ==1  \
                else ((P_c_a+P_c_d+ P_rx)*T_RF +P_trans*T_trans)/ E_max # normalized energy consumption in case active camera
            E_norm_s = (P_c_s+ P_rx)*T_RF / E_max if camera_state[t-1,1] ==1  \
                else ((P_c_s+ P_rx)*T_RF +P_trans*T_trans)/ E_max  # normalized energy consumption in case sleep camera
            alpha = 1.0
            J_a = E_pD[t] - alpha* E_norm_a # cost function in case active camera
            J_s = - alpha* E_norm_s # cost function in case sleep camera
            if J_a > J_s: 
                indices_gp[0,MCtest_idx] += nb_detections
                indices_gp[1,MCtest_idx] += fps-nb_detections
                E = (P_c_a+P_c_d+ P_rx)*T_RF
                CE_gp[t,MCtest_idx] = CE_gp[t-1,MCtest_idx] + E*((fps-nb_detections)/fps)
                E_trans = 0 if camera_state[t-1,1] ==1  else P_trans*T_trans
                camera_state[t,1] = 1
            else:
                indices_gp[2,MCtest_idx] += fps-nb_detections
                indices_gp[3,MCtest_idx] += nb_detections
                E = (P_c_s +P_rx)*T_RF
                CE_gp[t,MCtest_idx] = CE_gp[t-1,MCtest_idx] +  E*((nb_detections)/fps)
                E_trans = 0 if camera_state[t-1,1] ==0  else P_trans*T_trans
                camera_state[t,1]=0
            # "wrong" energy consumption for proposed approach
            CE_gp[t,MCtest_idx] += E_trans
            E_tot[2,MCtest_idx] += (E+E_trans)

            # camera active according to groundtruth POD
            E_norm_a = (P_c_a+P_c_d+ P_rx)*T_RF / E_max if camera_state[t-1,2] ==1  \
                else ((P_c_a+P_c_d+ P_rx)*T_RF +P_trans*T_trans)/ E_max
            E_norm_s = (P_c_s+ P_rx)*T_RF / E_max if camera_state[t-1,2] ==1  \
                else ((P_c_s+ P_rx)*T_RF +P_trans*T_trans)/ E_max
            J_a = pD_true_test[t] - alpha* E_norm_a
            J_s = - alpha* E_norm_s
            if J_a > J_s:
                indices_groundtruth[0,MCtest_idx] += nb_detections
                indices_groundtruth[1,MCtest_idx] += fps-nb_detections
                E = (P_c_a+P_c_d+ P_rx)*T_RF
                CE_groundtruth[t,MCtest_idx] = CE_groundtruth[t-1,MCtest_idx] + E*((fps-nb_detections)/fps)
                E_trans = 0 if camera_state[t-1,2] ==1  else P_trans*T_trans
                camera_state[t,2]=1
            else:
                indices_groundtruth[2,MCtest_idx] += fps-nb_detections
                indices_groundtruth[3,MCtest_idx] += nb_detections
                E = (P_c_s +P_rx)*T_RF
                CE_groundtruth[t,MCtest_idx] = CE_groundtruth[t-1,MCtest_idx] +  E*((nb_detections)/fps)
                E_trans = 0 if camera_state[t-1,2] ==0  else P_trans*T_trans 
                camera_state[t,2]=0
            # "wrong" energy consumption for groundtruth
            CE_groundtruth[t,MCtest_idx] += E_trans
            E_tot[3,MCtest_idx] += (E+E_trans)
        
        # normalize the confusion energies
        CE_always[-1,MCtest_idx] = CE_always[-1,MCtest_idx]/E_tot[0,MCtest_idx]
        CE_random[-1,MCtest_idx] = CE_random[-1,MCtest_idx]/E_tot[1,MCtest_idx]
        CE_gp[-1,MCtest_idx] = CE_gp[-1,MCtest_idx]/E_tot[2,MCtest_idx]
        CE_groundtruth[-1,MCtest_idx] = CE_groundtruth[-1,MCtest_idx]/E_tot[3,MCtest_idx]
        # compute Hamming distances
        ham_always[MCtest_idx] = distance.hamming(np.ones(N_test),camera_state[:,2])
        ham_random[MCtest_idx] = distance.hamming(camera_state[:,0],camera_state[:,2])
        ham_gp[MCtest_idx] = distance.hamming(camera_state[:,1],camera_state[:,2]) 

                    # *** RESULTS ANALYSIS ***

indices_list = [indices_always,indices_random,indices_gp,indices_groundtruth]
names_list = ["always", "rnd","GP","gt"]
colors = ['c','y','g','b']
# accuracy ecdf
fig =plt.figure(figsize=(9,6))
for i in range(len(indices_list)):
    acc, ecdf_acc = accuracy(indices_list[i])
    plt.plot(ecdf_acc.x, ecdf_acc.y, \
            linewidth=2,c=colors[i],label=names_list[i])
    # plt.title("Accuracy ECDF")
    plt.xlabel(r"$a$",fontsize=35)
    plt.ylabel(r"$p(acc \leq a)$",fontsize=35)
plt.legend(fontsize=30)
plt.xticks(np.arange(0, 1.01, step=0.2),fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

fig =plt.figure(figsize=(9,6))
for i in range(len(indices_list)):
    ecdf_Etot = ECDF(E_tot[i,:])
    plt.plot(ecdf_Etot.x, ecdf_Etot.y, \
            linewidth=2,c=colors[i],label=names_list[i])
    # plt.title("Energy ECDF")
    plt.xlabel(r"$E\;[J]$",fontsize=35)
    plt.ylabel(r"$p(E_{T_W} \leq E)$",fontsize=35)
plt.legend(fontsize=30)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(9,6))
CE_list = [CE_always,CE_random,CE_gp,CE_groundtruth]
time = range(N_test)
for i in range(len(CE_list)):
    # ax = plt.gca()
    # mean_E = np.mean(CE_list[i],axis=1)
    # plt.plot(time,mean_E,linestyle='-',label=names_list[i],linewidth=2,c=colors[i])
    # std_E = np.std(CE_list[i],axis=1)
    # plt.plot(time,mean_E-std_E,linestyle='--',linewidth=1,c=colors[i])
    # plt.plot(time,mean_E+std_E,linestyle='--',linewidth=1,c=colors[i])
    # ax.fill_between(time, mean_E-std_E, mean_E+std_E ,alpha=0.2, facecolor=colors[i])
    ecdf_E = ECDF(CE_list[i][-1,:])
    plt.plot(ecdf_E.x, ecdf_E.y, \
            linewidth=2,c=colors[i],label=names_list[i])
    # plt.title("Energy ECDF")
    plt.xlabel(r"$E$",fontsize=35)
    plt.ylabel(r"$p(CE_{T_W} \leq E)$",fontsize=35)
plt.legend(fontsize=30,framealpha=0.5,loc='lower right')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

for i in range(np.shape(E_tot)[0]):
    print(np.mean(E_tot[i,:]), np.std(E_tot[i,:]))

# plot the last MC test
print(np.mean(ham_gp), np.std(ham_gp), ham_gp[-1])
fig = plt.figure(figsize=(9,6))
idx_GP = 0
idx_gt = 0
label_on_GP =0
label_on_gt =0
plt.plot(abs(E_pD-pD_true_test),c='b',linestyle=':',linewidth=0.5)
for i in range(N_test):
    if camera_state[i,1]!=camera_state[i,2]:
        plt.plot(i,abs(E_pD[i]-pD_true_test[i]), alpha=0.3,marker='o',markersize=5.0,c='b', label=r"$s_t^{(gt)} \neq s_t^{(GP)}$")
    else: 
        plt.plot(i,abs(E_pD[i]-pD_true_test[i]), alpha=1,marker='o',markersize=5.0,c='b', label=r"$s_t^{(gt)} = s_t^{(GP)}$")
    # if (i > 0 and camera_state[i-1,1]+camera_state[i,1]==1) or i==N_test-1:
    #     if camera_state[i-1,1]==0:
    #         plt.plot(range(max(idx_GP-1,0),i), E_pD[max(idx_GP-1,0):i], alpha=0.5 ,linewidth=1,c=colors[-2])
    #     else:
    #         if not label_on_GP:
    #             plt.plot(range(max(idx_GP-1,0),i), E_pD[max(idx_GP-1,0):i], alpha=1 ,linewidth=1.5,c=colors[-2],label='GP')
    #             label_on_GP +=1
    #         else: 
    #             plt.plot(range(max(idx_GP-1,0),i), E_pD[max(idx_GP-1,0):i], alpha=1 ,linewidth=1.5,c=colors[-2])
    #     idx_GP = i
    # if (i > 0 and camera_state[i-1,2]+camera_state[i,2]==1) or i==N_test-1:
    #     if camera_state[i-1,2]==0:
    #         plt.plot(range(max(idx_gt-1,0),i), pD_true_test[max(idx_gt-1,0):i], alpha=0.5 ,linewidth=1.5,c=colors[-1])
    #     else:
    #         if not label_on_gt:
    #             plt.plot(range(max(idx_gt-1,0),i), pD_true_test[max(idx_gt-1,0):i], alpha=1 ,linewidth=1.5,c=colors[-1],label='gt')
    #             label_on_gt+=1
    #         else: 
    #             plt.plot(range(max(idx_gt-1,0),i), pD_true_test[max(idx_gt-1,0):i], alpha=1 ,linewidth=1.5,c=colors[-1])
    #     idx_gt = i
plt.legend(fontsize=30,framealpha=0.5,loc='upper right',ncol=2,columnspacing=0.1)
plt.xlabel(r'$t\;[s]$',fontsize=35)
plt.ylabel(r'$|J_D^{(gt)} - J_D^{(GP)}|$',fontsize=35)
plt.ylim([0,1.4])
plt.tight_layout()
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

# hist of Hamming distances
fig = plt.figure(figsize=(9,6))
plt.hist(ham_always,density=True, color=colors[0],alpha=0.5, label=names_list[0])
plt.hist(ham_random,density=True, color=colors[1],alpha=0.5, label=names_list[1])
plt.hist(ham_gp,density=True, color=colors[2],alpha=0.5, label=names_list[2])
plt.xlabel(r'$d_H(\cdot,\cdot)$',fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.legend(fontsize=30)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()
