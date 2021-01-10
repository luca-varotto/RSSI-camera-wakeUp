import os 
from datetime import datetime

############################################################################################################################

#  COLLECT RSSI

############################################################################################################################

dataset_type = "test"
prefix = datetime.now().strftime("%m%d%Y")
output_path = './data/' + prefix 

# Start RSSI collection
os.system('./utils/JLinkRTTLogger -Device NRF52832_XXAA -If swd -Speed 4000 -RTTChannel 4000 -RTTChannel 0 ' + output_path+"-rssi-"+ dataset_type +".csv") 