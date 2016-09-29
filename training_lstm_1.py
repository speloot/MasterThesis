import keras
import time
import h5py
import os.path
import numpy as np
from keras import callbacks
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, History
import random 
# --------------------------------------------------------------------
# Set the random number generators for consistency
seed = 123
np.random.seed(seed)

# Network Parameters
nNeurons = 25   # number of hidden neurons
nEpoch   = 15   # Max. number of Epochs
sBatch   = 1    # Batch Size
dOut     = 0.05 # DropOut Ratio

#-------------result_fileName------------------------------------------
modelName  = "lstm_nadam"
folderName = "%s_%d_%d_%d" %(modelName, nNeurons, nEpoch, sBatch)
dirName    = "%s_%d_%d_%d" %(modelName, nNeurons, nEpoch, sBatch)

#-------------Mac/Lin--------------------------------------------------
macPath   = '/Users/..'
# Create a directory to save the results in
destPath = os.path.join(macPath, dirName ) 
if not os.path.exists(destPath):
    os.makedirs(destPath)

#-------------result_filePath-----------------------------------------
logPath    = os.path.join(destPath, folderName + "_log.txt" ) 
weightPath = os.path.join(destPath, folderName + "_weights.hdf5" ) 
modelPath  = os.path.join(destPath, folderName + "_my_model.h5") 

# --------------------------------------------------------------------
# --------------------------Data PreProcessing------------------------
# --------------------------------------------------------------------
#------------------------------- Load DATA----------------------------
print('Loading DATA...')
dataFile = open('DS_US_N_MM.txt','r')
df       = np.loadtxt(dataFile, delimiter = ',',usecols = (0,1,2,3,4))
dataFile.close

#------------------------------Shuffle DATA---------------------------
np.random.shuffle(df)

#------------------------------Split DATA-----------------------------
# Retrieve DATA
(X_train, Y_train), (X_valid, Y_valid) = train_valid_split(df)

#---------------------------
print(len(X_train), ' train sequences')
print(len(X_valid), ' valid sequences')
print(nNeurons, ' Neurons')
print(sBatch, ' Batch Size')
print(nEpoch, ' Iteration')



# Reshape DATA for RNN
#-------------Reshape Data--------------------------------------------
X_train  = np.reshape(X_train, X_train.shape + (1,)) 
X_valid  = np.reshape(X_valid, X_valid.shape + (1,))
# --------------------------------------------------------------------
# --------------------------NN----------------------------------------
# --------------------------------------------------------------------
# Model  Architecture
print('Building the model...')
tic = time.time()

model = Sequential()
model.add(LSTM(input_shape      = (X_train.shape[0], 1),
              batch_input_shape = (sBatch, X_train.shape[0], 1),
              output_dim        = nNeurons,
              init              ='he_uniform',
              activation        = 'sigmoid'
              )
          )

model.add(Dropout(dOut))

#sgd      = keras.optimizers.SGD(lr=0.01, momentum=0.1, decay=0.1, nesterov=True)
#adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
nadam = keras.optimizers.Nadam(lr    = 0.002,
                              beta_1 = 0.9,
                              beta_2 = 0.999,
                              epsilon= 1e-08,
                              schedule_decay=0.004)

model.add(Dense(1,))

model.compile(loss='mean_squared_error', optimizer='nadam')
print('Training the model...')

#------------------Train the Network--------------------------------------

checkpointer = ModelCheckpoint(weightPath, verbose=2, save_best_only=True)

history = model.fit(X_train, Y_train,
                    nb_epoch=nEpoch,
                    batch_size=sBatch,
                    validation_data = (X_valid, Y_valid),
                    shuffle=True,   # shuffle after each Epoch!
                    callbacks = [checkpointer]
		                )
# Save the model
model.save(modelPath  ) 
# Early Stopping 
keras.callbacks.EarlyStopping(monitor='val_loss'
							, patience=1
							, verbose=1
							, mode='min')


#------------------Evaluate The Model--------------------------------------
score = model.evaluate(X_valid, Y_valid, batch_size = sBatch)
# print('score: ', score)


#-------------------Model Config & Summary---------------------------------
modelSummary = model.summary()
modelConfig = model.get_config()

#----------------------Track time!-----------------------------------------
elapsedTime = (time.time()-tic) / 60

# Write results to file
result_file = open(logPath,"w")
result_file.write('Train Sequences: ' + repr(X_train_len)  +'\n')
result_file.write('Test Sequences:  ' + repr(X_valid_len)  +'\n')
result_file.write('Batch Size =     ' + repr(sBatch)       +'\n')
result_file.write('# Epoch    =     ' + repr(nEpoch)       +'\n')
result_file.write('# neuron_1 =     ' + repr(hid_layer_1)  +'\n')
result_file.write('Mode Config:     ' + repr(modelConfig)  +'\n')
result_file.write(' '+'\n')
result_file.write('Training Loss:   ' +repr(history.history['loss'])    +'\n')
result_file.write(' '+'\n')
result_file.write('Validation Loss: ' +repr(history.history['val_loss'])+'\n')
result_file.write(' '+'\n')
result_file.write('Model Score:     ' +repr(score)         + '\n')
result_file.write('Elapsed time:    ' +repr(elapsedTime)   + ' min'     +'\n')
result_file.close()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_valid_split(df, valid_size=0.07513148009016):  # calculated for sBatch=100
    """
    This just splits data to training and validating parts
    """
    ntrn = round(len(df) * (1 - valid_size))

    X_train = df[0:ntrn,0:4]
    Y_train = df[0:ntrn,4]
    X_valid  = df[ntrn:,0:4]
    Y_valid  = df[ntrn:,4]

    return (X_train, Y_train), (X_valid, Y_valid)