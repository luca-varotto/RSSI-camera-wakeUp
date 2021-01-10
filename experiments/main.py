from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
import pickle
from scipy.misc import derivative
import cv2
from tqdm import tqdm

############################################################################################################################

from pf import PF

############################################################################################################################

# function to predict the POD, given the GP model and a RSSI value 
def f(x,model):
    return model.predict(np.array([x]).reshape(-1,1))[0][0]

############################################################################################################################

# USE DATASET CONTAINING COUPLES (pD_meas,rssi) TO TRAIN A GP MODEL AND SOLVE THE CAMERA WAKE-UP PROBLEM

############################################################################################################################

                    # *** SETUP PARAMETERS ***
# RSSI tracking via PF
N_s = 100 # number of particles
mu_omega = 0.0 # mean of the process model
sigma_omega = 0.1 # std. dev. of the process model 
pf_train = PF(N_s,\
            init_interval=[-100,-10],\
            draw_particles_flag=False)

# devices specifications
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

# LOAD TRAING DATA
data_csv = "11242020-data-train.csv"
data = pd.read_csv('./data/'+data_csv,header=None)
# pD_meas
pD_meas = data.iloc[:, 1].to_numpy().reshape(-1,1) 
# rssi 
rssi = data.iloc[:, 2].to_numpy().reshape(-1,1)

print("GP training on {:d} samples".format(len(rssi)))

# GP MODEL TRAINING
# define GP model
model_gp = GaussianProcessRegressor(Matern()+WhiteKernel(noise_level_bounds=(0.0,0.2)))
# fit the model
model_gp.fit(rssi,pD_meas)
idx = np.argsort(rssi,axis=0)
pD_est_gp,std_gp = model_gp.predict(rssi[idx,0],return_std=True)
pD_est_gp = np.clip(pD_est_gp,0.0,1.0)

# define NIGP model
f_der = np.zeros(len(rssi))
for i in range(len(rssi)):
    f_der[i] = derivative(f,rssi[i],args=(model_gp,))
model_nigp = GaussianProcessRegressor(kernel=\
    Matern(length_scale=np.exp(model_gp.kernel_.theta)[0],length_scale_bounds="fixed") + \
    WhiteKernel(noise_level=np.exp(model_gp.kernel_.theta)[1],noise_level_bounds=(0.0,0.2))+\
    WhiteKernel(),alpha=f_der**2)
# fit the NIGP model
model_nigp.fit(rssi,pD_meas)
pD_est_nigp,std_nigp = model_nigp.predict(rssi[idx,0],return_std=True)
pD_est_nigp = np.clip(pD_est_nigp,0.0,1.0)

# plot training results
fig = plt.figure(figsize=(9,6))
# plt.subplot(2,1,1)
# plt.plot(rssi[idx,0], pD_est_gp,c='g',label='$\widehat{p}_D(r)$',linewidth=2)
# plt.plot(rssi[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
#     alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
# plt.fill_between(np.squeeze(rssi[idx,0]),\
#         np.squeeze(pD_est_gp) - std_gp,\
#         np.squeeze(pD_est_gp) + std_gp,\
#         alpha=0.5, facecolor='g') 
# plt.legend(fontsize=30)
# plt.xlabel(r'$r$ [dBm]',fontsize=35)
# # plt.fill_between(np.squeeze(rssi[idx,0]),\
# #         np.squeeze(pD_est_gp) - (std_gp + np.squeeze(f_der[idx]**2)),\
# #         np.squeeze(pD_est_gp) + (std_gp+ np.squeeze(f_der[idx]**2)),\
# #         alpha=0.2, facecolor='g', label='CI + noise')
# print('LML: {:5.3f}'.format(model_gp.log_marginal_likelihood(model_gp.kernel_.theta)) + ', '\
#     'R^2'+': {:4.3f}'.format(model_gp.score(rssi[idx,0],pD_ideal)) + ', '\
#     'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_gp)))
#     )
# plt.subplot(2,1,2)
plt.plot(rssi[idx,0], pD_est_nigp,c='g',label='$\widehat{p}_D(r)$',linewidth=2) # estimated RSSI-POD function
plt.plot(rssi[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
    alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
plt.fill_between(np.squeeze(rssi[idx,0]),\
        np.squeeze(pD_est_nigp) - std_nigp,\
        np.squeeze(pD_est_nigp) + std_nigp,\
        alpha=0.5, facecolor='g') 
plt.legend(fontsize=30)
plt.xlabel(r'$r\;[dBm]$',fontsize=35)
# print('LML: {:5.3f}'.format(model_nigp.log_marginal_likelihood(model_nigp.kernel_.theta)) + ', '\
#     'R^2'+': {:4.3f}'.format(model_nigp.score(rssi[idx,0],pD_ideal)) + ', '\
#     'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_nigp)))
#         )
plt.xticks(np.arange(int(min(rssi)), int(max(rssi)), step=int((int(max(rssi))-int(min(rssi)))/5)),fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

                    # *** GP TESTING *** 

# LOAD TEST DATA
data_test_pkl = "./data/11242020-data-test.pkl"
file_to_read = open(data_test_pkl, "rb")
data_test = pickle.load(file_to_read)
detections_per_sample = []
rssi_test = []
for key, value in data_test.items():
    detections_per_sample.append(value[:-1][0])
    rssi_test.append(value[-1])

time_stmps = list(data_test)
T_RF = ( int(time_stmps[-1])*1.0E-3 - int(time_stmps[0])*1.0E-3 ) /  len(time_stmps) # [s] average Rx sampling time 

print("GP test on {:d} samples".format(len(rssi_test)))


# PF for RSSI tracking
pf_test = PF(N_s,\
    init_interval=[-100,-10],\
    draw_particles_flag=False)

# show video
cap = cv2.VideoCapture("./data/11242020-video.avi")

# performance indices: TP, FP. TN, FN (for each MC test) 
indices_always = np.zeros(4) # camera always active
indices_random = np.zeros(4) # camera randomly active
indices_gp = np.zeros(4) # proposed GP-based camera activation
# energy consumption
CE_always = np.zeros(len(rssi_test))
CE_random = np.zeros(len(rssi_test))
CE_gp = np.zeros(len(rssi_test))
E_tot = np.zeros(4)

rssi_smooth_test = np.empty(len(rssi_test))
E_pD = np.empty(len(rssi_test))
camera_state = np.zeros((len(rssi_test),3),dtype=int) # random, GP, groundtruth

# throw away the frames of the first rssi sample
for _ in range(len(detections_per_sample[0])):
    ret, frame = cap.read() 


for t in tqdm(range(1,len(rssi_test))): 

    # RSSI update
    pf_test.update(rssi_test[t-1],3)
    # resampling
    pf_test.SIS_resampling()
    # compute MMSE estimate
    pf_test.estimation(est_type='MMSE')
    rssi_smooth_test[t-1] = pf_test.estimate

    # number of frames for the next RSSI sample
    fps =  len(detections_per_sample[t])

    # RSSI predict
    pf_test.predict(mu_omega,sigma_omega)
    pD_predict = 1-(1- model_nigp.predict(pf_test.particles.reshape(-1,1),return_std=False))**fps
    E_pD[t] = np.average(pD_predict, weights=pf_test.weights, axis=0)

    # number of detections if the camera is switched on 
    nb_detections = np.sum(detections_per_sample[t])

    # indices for camera always active
    indices_always[0] += nb_detections
    indices_always[1] += fps-nb_detections
    # "wrong" energy consumption for camera always active
    E = (P_c_a +P_c_d)*T_RF
    E_tot[0] += E
    CE_always[t] = CE_always[t-1] + E*((fps-nb_detections)/fps)


    # camera randomly active 
    if np.random.binomial(1, 0.5): # activate according to a coin toss 
        indices_random[0] += nb_detections
        indices_random[1] += fps-nb_detections
        E = (P_c_a +P_c_d)*T_RF
        CE_random[t] = CE_random[t-1] + E*((fps-nb_detections)/fps)
        E_trans = 0 if camera_state[t-1,0] ==1 else P_trans*T_trans
        camera_state[t,0]=1
    else:
        indices_random[2] += fps-nb_detections
        indices_random[3] += nb_detections
        E = P_c_s*T_RF
        CE_random[t] = CE_random[t-1] +  E*((nb_detections)/fps)
        E_trans = 0 if camera_state[t-1,0] ==0 else P_trans*T_trans
        camera_state[t,0]=0
    # "wrong" energy consumption for camera randomly active
    CE_random[t] += E_trans
    E_tot[1] += (E+E_trans)
    
    E_max = (P_c_a+P_c_d+ P_rx)*T_RF + P_trans*T_trans # max energy consumption
    E_norm_a = (P_c_a+P_c_d+ P_rx)*T_RF / E_max if camera_state[t-1,1] ==1  \
        else ((P_c_a+P_c_d+ P_rx)*T_RF +P_trans*T_trans)/ E_max # normalized energy consumption in case active camera
    E_norm_s = (P_c_s+ P_rx)*T_RF / E_max if camera_state[t-1,1] ==1  \
        else ((P_c_s+ P_rx)*T_RF +P_trans*T_trans)/ E_max  # normalized energy consumption in case sleep camera
    alpha = 1.0
    J_a = E_pD[t] - alpha* E_norm_a # cost function in case active camera
    J_s = - alpha* E_norm_s # cost function in case sleep camera
    if J_a > J_s: 
        indices_gp[0] += nb_detections
        indices_gp[1] += fps-nb_detections
        E = (P_c_a+P_c_d+ P_rx)*T_RF
        CE_gp[t] = CE_gp[t-1] + E*((fps-nb_detections)/fps)
        E_trans = 0 if camera_state[t-1,1] ==1  else P_trans*T_trans
        camera_state[t,1] = 1
    else:
        indices_gp[2] += fps-nb_detections
        indices_gp[3] += nb_detections
        E = (P_c_s +P_rx)*T_RF
        CE_gp[t] = CE_gp[t-1] +  E*((nb_detections)/fps)
        E_trans = 0 if camera_state[t-1,1] ==0  else P_trans*T_trans
        camera_state[t,1]=0
    # "wrong" energy consumption for proposed approach
    CE_gp[t] += E_trans
    E_tot[2] += (E+E_trans)

    # for f in range(fps):
    #     ret, frame = cap.read()
    #     if camera_state[t,1]==1:
    #         frame = cv2.circle(frame, (20,20), 10, (0,255,0), -1)
    #     else: 
    #         frame = cv2.circle(frame, (20,20), 10, (0,0,255), -1)
        # if active==1 and detections_per_sample[t][f]==1: 
        #     cv2.imwrite("./data/TP.jpg",frame)
        # if active==0 and detections_per_sample[t][f]==1:
        #     cv2.imwrite("./data/FN.jpg",frame)
        # if active==1 and detections_per_sample[t][f]==0: 
        #     cv2.imwrite("./data/FP.jpg",frame)
        # if active==0 and detections_per_sample[t][f]==0:
        #     cv2.imwrite("./data/TN.jpg",frame)
        # cv2.imshow('frame',frame)
        # close the stream video
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# cv2.destroyAllWindows()

# normalize the confusion energies
CE_always[-1] = CE_always[-1]/E_tot[0]
CE_random[-1] = CE_random[-1]/E_tot[1]
CE_gp[-1] = CE_gp[-1]/E_tot[2]

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(rssi_test,label=r'$\tilde{r}$')
plt.plot(rssi_smooth_test[:-1],label=r'$\tilde{r}^{PF}$')
plt.xlabel(r'$t$')
plt.ylabel(r'$RSSI$')
plt.legend()
plt.subplot(2,1,2)
plt.plot(E_pD[1:],label=r'$\mathbb{E}[p_D]$')
plt.plot(camera_state[:,1],label='active', linestyle=':', marker='o',markersize=1.0,c='k',alpha=0.5)
plt.legend()
plt.xlabel(r'$t$')
plt.ylim([0,1.01])
plt.tight_layout()
plt.show()

indices_list = [indices_always,indices_random,indices_gp]
E_list = [CE_always,CE_random,CE_gp]
names_list = ["always", "random","GP"]
# accuracy ecdf
fig =plt.figure()
for i in range(len(indices_list)):
    print("Acc " + names_list[i] + ": {:4.3f}".format( (indices_list[i][0] + indices_list[i][2])/ np.sum( np.array(indices_list[i]))))
    print("CE " + names_list[i] + ": {:4.3f}".format( E_list[i][-1]) + " J" )
    print("E tot " + names_list[i] + ": {:4.3f}".format( E_tot[i]) + " J" )