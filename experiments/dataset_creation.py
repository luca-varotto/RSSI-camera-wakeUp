import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

############################################################################################################################

from kalman_filter import Kalman_filter
from pf import PF

############################################################################################################################

# SYNCHRONIZE AND FUSE CAMERA AND RSSI DATA

############################################################################################################################

############################################################################################################################

from kalman_filter import Kalman_filter

############################################################################################################################


# specify the type of dataset: train/test
dataset_type = "train"

# RSSI
rssi_csv_name = "11242020-rssi-"+dataset_type+".csv"
line_start = 16 # rssi data before this line must be removed (N.B.: line counter starts with 1)
rssi_df = pd.read_csv('./data/'+rssi_csv_name,header=None)
# rssi samples time (first sample supposed to be collected instantaneously after the program launch) [ms]
rssi_time = (rssi_df.iloc[line_start-1:, 0].to_numpy() - rssi_df.iloc[line_start-1, 0])*10
# rssi sample values [dBm] 
rssi = rssi_df.iloc[line_start-1:, 2].to_numpy()

# estimate Fs (sampling frequency)
Fs = len(rssi_time)/( rssi_time[-1]/1000 )
print("Fs: " + str( Fs )  + " [Hz]" )

# RSSI smoothing by PF tracking [find a suitable process model wrt RSSI according to the process model wrt distance]
N_s = 100 # number of particles
mu_omega = 0.0 # mean of the process model
sigma_omega = 0.1 # std_gp. dev. of the process model 
pf = PF(N_s,\
        init_interval=[-100,-10],\
        draw_particles_flag=False)
rssi_smooth = np.empty(np.shape(rssi))
for t in range(len(rssi_time)):
    # RSSI predict
    pf.predict(mu_omega,sigma_omega)
    # RSSI update
    pf.update(rssi[t],3)
    # resampling
    pf.SIS_resampling()
    # compute target position estimate
    pf.estimation(est_type='MMSE')
    rssi_smooth[t] = pf.estimate

# AVERAGED RSSI
# average groups of Fs consecutive data, so that the new Fs is 1 [Hz]
Fs_int = int(Fs/Fs)
rssi_avg = []
rssi_avg_time = []
i = 0 
while i < len(rssi_smooth)-Fs_int:
    rssi_avg.append( np.mean( rssi_smooth[i:i+Fs_int] ) )
    rssi_avg_time.append( rssi_time[i+Fs_int-1] )
    i += Fs_int

# estimate Fs (sampling frequency)
Fs_avg = len(rssi_avg_time)/( rssi_avg_time[-1]/1000 )
print("Fs_avg: " + str( Fs_avg )  + " [Hz]" )

# CAMERA
camera_csv_name = "11242020-camera-"+dataset_type+".csv"
camera_df = pd.read_csv('./data/'+camera_csv_name)
# camera frames time [ms]
frame_time = camera_df['timestmp'].to_numpy()
# camera detections 
detections = camera_df['detection'].to_numpy()

# estimate FPS
fps = len(frame_time)/( (frame_time[-1]-frame_time[0])/1000 ) 
print("FPS: " + str( fps ) + " [Hz]")

# FRAME-RSSI ASSOCIATION
# Kalman filter to smooth POD measurements
x0 = 0.5 # initial guess
kf = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1) # A, B, C, Q, R, P
if fps > Fs: # we need to associate at least one frame per each rssi sample
    
    # associate to each rssi sample the estimated pD
    frame_idx = 0
    data = {}
    for i in range(len(rssi_avg_time)):
        frames_per_sample = []
        t_rssi = rssi_avg_time[i] 
        while frame_idx < len(frame_time) and frame_time[frame_idx] <= t_rssi: 
                frames_per_sample.append(frame_time[frame_idx])
                frame_idx += 1
        if len(frames_per_sample) > 0:
            if dataset_type == 'train':
                pD_est = 0 
                for j in frames_per_sample:
                    pD_est += detections[np.where(frame_time == j)]
                pD_est = pD_est.item()/len(frames_per_sample)
                kf.predict()
                kf.update(pD_est)
                data[str(t_rssi)] = [kf.x,rssi_avg[i]] 
            else:
                detections_per_sample = [] 
                for j in frames_per_sample:
                    detections_per_sample.append( detections[np.where(frame_time == j)].item() )
                data[str(t_rssi)] = [detections_per_sample,rssi_avg[i]]
            

# plot [highlight the different Ts of RSSI and camera]
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(rssi_time,rssi,label="raw") # raw RSSI
plt.plot(rssi_time,rssi_smooth,label="PF smooth") # smooth RSSI
# plt.plot(rssi_avg_time,rssi_avg,label='averaged') # averaged RSSI
plt.legend()
plt.title("RSSI")
plt.xlabel("t [ms]")
plt.xlim([0,max(rssi_time)])
plt.subplot(2,1,2)
plt.plot(frame_time,detections, label="camera detections") # camera detections
# ........ plot measured pD
plt.title("camera")
plt.xlabel("t [ms]")
plt.xlim([0,max(frame_time)])
plt.tight_layout()
plt.show()


# SAVE DATA
if dataset_type == 'train':
    csv_file = './data/' + rssi_csv_name[:8] + "-data-" +  dataset_type + ".csv"
    csv_columns = ["timestmp","pD_est","rssi_avg"] 
    with open(csv_file, 'w') as f:
        for key in data.keys():
            f.write("%s,%s,%s\n"%(key,data[key][0],data[key][1]))
else:
    pickle_file = './data/' + rssi_csv_name[:8] + "-data-" +  dataset_type + ".pkl"
    f = open(pickle_file,"wb")
    pickle.dump(data,f)
    f.close()


    



