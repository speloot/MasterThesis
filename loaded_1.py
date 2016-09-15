import keras
import time
import h5py
import os.path
import numpy as np
from keras import callbacks
from keras.optimizers import Nadam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l1, ActivityRegularizer
from keras.callbacks import ModelCheckpoint, History
from keras.models import load_model
# --------------------------------------------------------------------
# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)
# Determin Network Parameters
hid_layer_1 = 25
nEpoch = 15
sBatch = 1
model_name = "mlp_nadam"
#-------------result_fileName------------------------------------------
result_txt_Name = "%s_%d_%d_%d" %(model_name,hid_layer_1, nEpoch,sBatch)
dirName = "%s_%d_%d_%d" % (model_name, hid_layer_1, nEpoch, sBatch)
#-------------Mac/Lin--------------------------------------------------
macPath   = '/Users/SiamakEsmi/Desktop/final_cpu_model/02'
# Create a dir for saving the result in
destPath = os.path.join(macPath, dirName ) 
if not os.path.exists(destPath):
    os.makedirs(destPath)
#-------------result_filePath-----------------------------------------
log_path = os.path.join(destPath, result_txt_Name + "_logFile.txt" ) 
weight_path = os.path.join(destPath, result_txt_Name + "_weights.hdf5" ) 
final_pred_path = os.path.join(destPath, result_txt_Name + "_final_pred.txt" ) 
model_path = os.path.join(destPath, result_txt_Name + "_my_model.h5") 
data_path = os.path.join(destPath, result_txt_Name + "_data_df_copy.txt" ) 

print('Loading DATA...')
dataFile = open('DS_US_N_MM.txt','r')
df_copy = np.loadtxt(dataFile, delimiter = ',',usecols = (0,1,2,3,4))
dataFile.close


model = Sequential()
loadedModel = load_model(model_path)
print('Model is loadd..')
loadedModel.load_weights(weight_path)
loadedModel.compile(loss='mean_squared_error', optimizer="nadam")
predicted_df_copy = loadedModel.predict(df_copy[:,0:4], batch_size=1)
np.savetxt(final_pred_path,predicted_df_copy) 
np.savetxt(data_path, df_copy[:,4])