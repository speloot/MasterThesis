import keras
import h5py
import numpy as np
from keras.models import Sequential
from keras.models import load_model
# --------------------------------------------------------------------

# Determin Network Parameters
hid_layer = 25
sBatch = 1
model_name = "modelName"
# Define tha result path and fileName
#-------------result_fileName------------------------------------------

#-------------result_filePath-----------------------------------------

print('Loading DATA...')
dataFile = open('dataFile.txt','r')
df_copy = np.loadtxt(dataFile, delimiter = ',',usecols = (0,1,2,3,4))
dataFile.close

model = Sequential()
loadedModel = load_model(model_path)
print('Model is loaded')
loadedModel.load_weights(weight_path)
print('Weights are loaded')
loadedModel.compile(loss='mean_squared_error', optimizer="nadam")
print('Model is compiled')
predicted_df_copy = loadedModel.predict(df_copy[:,0:4], batch_size=1)
np.savetxt(final_pred_path,predicted_df_copy) 
np.savetxt(data_path, df_copy[:,4])
