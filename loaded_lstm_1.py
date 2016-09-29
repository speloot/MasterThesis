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
# Set the random number generators for consistency
seed = 123
np.random.seed(seed)

# Network Parameters
nNeurons = 25	# number of hidden neurons
nEpoch = 15		# Max. number of Epochs
sBatch = 1 		# Batch Size

#-------------result_fileName------------------------------------------
modelName	 = "lstm_nadam"
folderName	 = "%s_%d_%d_%d" %(modelName, nNeurons, nEpoch, sBatch)
dirName		 = "%s_%d_%d_%d" %(modelName, nNeurons, nEpoch, sBatch)

#-------------Mac/Lin--------------------------------------------------
macPath   = '/Users/..'
# Create a directory to save the results in
destPath = os.path.join(macPath, dirName ) 
if not os.path.exists(destPath):
    os.makedirs(destPath)

#-------------Result filePath-----------------------------------------
log_path 	= os.path.join(destPath, folderName	+ "_logFile.txt") 
weightPath 	= os.path.join(destPath, folderName	+ "_weights.hdf5") 
predPath 	= os.path.join(destPath, folderName + "_predicted.txt") 
modelPath	= os.path.join(destPath, folderName	+ "_my_model.h5") 
dataPath 	= os.path.join(destPath, folderName	+ "_data_df_copy.txt") 

#-------------Load Data-----------------------------------------------
print('Loading DATA...')
dataFile = open('DS_US_N_MM.txt','r')
df       = np.loadtxt(dataFile, delimiter = ',',usecols = (0,1,2,3,4))
dataFile.close
#-------------Reshape Data--------------------------------------------
df_copy = np.reshape(df, df.shape + (1,)) 

#-------------Laod the Saved Model------------------------------------
model 		= Sequential()
loadedModel = load_model(modelPath)
print('Model is loadd..')

#-------------Laod the Saved Weights----------------------------------
loadedModel.load_weights(weightPath)

#-------------Loss Function & Optimizer-------------------------------
loadedModel.compile(loss='mean_squared_error', optimizer="nadam")

#-------------Map Loaded Data-----------------------------------------
predicted_df_copy= oadedModel.predict(df_copy[:,0:4],batch_size=sBatch)

#-------------Save the Result-----------------------------------------
np.savetxt(predPath, predicted_df_copy) 
np.savetxt(dataPath, df_copy[:,4])