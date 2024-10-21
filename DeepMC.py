

import numpy as np
from keras.constraints import max_norm
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.layers import LSTM, Embedding, Conv1D, Concatenate, Dropout, GRU, Input, Dense, MaxPooling1D, Flatten, GaussianNoise
from keras.optimizers import Adam

# Function:
#   To apply dropout with optional Monte Carlo (MC) dropout
def get_dropout(input_tensor, p=0.25, mc=True):

    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)



if __name__ == '__main__':

   # Loading training  data
   print("READING TRAINING DATA")
   DATAFILE_Training = "data/TRAINING_Disimilarity_0_01.txt"
   f = open(DATAFILE_Training,'r')
   for line in f:
       header=str(line.split('\n')[0]).split("\r")[0].split("\t")
       break
   f.close()

   f = open(DATAFILE_Training,'r')
   i = 0
   X=[] # Empty list to store training data
   for line in f:
       if i>0: # Skip the header
           Temp = str(line.split('\n')[0]).split("\r")[0].split("\t")
           X.append(Temp)
       i=i+1
   f.close()

   # Convert training data to numpy array and separate features (X_train) and labels (y_train)
   X_train = np.array(X, dtype=float)
   y_train = X_train[:,0]
   X_train = X_train[:,1:]
   
   header = np.array(header)
   header = header[1:]

   # Loading external data
   print("READING EXTERNAL DATA")
   DATAFILE_External = "data/EXTERNAL_Disimilarity_0_01.txt"
   f = open(DATAFILE_External,'r')
   i = 0
   X=[]
   for line in f:
       if i>0: # Skip the header
           Temp = str(line.split('\n')[0]).split("\r")[0].split("\t")
           X.append(Temp)
       i=i+1
   f.close()

    # Convert external data to numpy array and separate features (X_train) and labels (y_train)
   X_test_Balanced = np.array(X, dtype=float)
   y_test_Balanced = X_test_Balanced[:,0]
   X_test_Balanced = X_test_Balanced[:,1:]  
   print(X_test_Balanced.shape)
   
   encoder_input_1 = X_train.shape[1:]  # Defining the input shape for the model
   
   # Build the model
   x1 = Input(shape=encoder_input_1)
   x = Dense(300, activation='relu')(x1)
   x = Dense(200, activation='relu')(x)
   x = get_dropout(x, p=0.25, mc=True)
   x = Dense(100, activation='relu')(x)
   x = Dense(50, activation='relu')(x)
   x = get_dropout(x, p=0.25, mc=True)
   x = Dense(1, activation='sigmoid')(x)
   
   model = Model(x1, x) # Creating the model
  
   print(model.summary()) # Print model summary
       
   # Compile the model with Adam optimizer and binary crossentropy loss
   opt=Adam(lr=0.001) #Default 0.001
   model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
   
   # Training the model
   model.fit(X_train,y_train,
                       epochs=50,
                       batch_size=20,
                       shuffle=True,
                       validation_data=(X_test_Balanced, y_test_Balanced))
   #Saving the model
   model.save("file_virus_model_MC_001.h5")
  
   # Run the model with other data
   print("RUNNING MODEL WITH OTHER DATA")
   DATAFILE_file_covid = "data/Sample_Virus_Disimilarity_0_01.txt"
   f = open(DATAFILE_file_covid,'r')
   i = 0

   X_covid=[]
   for line in f:
       if i>0:
           Temp = str(line.split('\n')[0]).split("\r")[0].split("\t")
           X_covid.append(Temp)
       i=i+1
   f.close()
   X_covid = np.array(X_covid, dtype=float)
   #X_covid = Xcovid.reshape(1, 365)
   print(X_covid.shape)

   # Load the saved model
   new_model = load_model('models/file_virus_model_MC_001.h5')

   # Prepare output files for predictions
   a_file = open("prediction_All_virus_Disimilarity_MC_0.001.txt", "w")
   b_file = open("prediction_All_virus_Disimilarity_MC_0.001_values.txt", "w")
   
   Predictions=[]

   for i in range(X_covid.shape[0]):
       XX = X_covid[i].reshape(1,-1)
       AP = []
       for j in range(200):
           AP.append(model.predict(XX, batch_size=1000)[0][0])
       b = [str(x) for x in AP]
       # Write the raw predictions to file
       b_file.write("\t".join(b)+"\n")
       AP = np.array(AP)
       Cl=0
       if np.mean(AP)>0.5: Cl=1
       # Calculate statistics for the predictions
       a = [Cl, np.mean(AP), np.median(AP), np.std(AP), np.min(AP), np.max(AP), np.std(AP)*100/ np.mean(AP), np.nanpercentile(AP, 2.5), np.nanpercentile(AP, 97.5), np.nanpercentile(AP, 25), np.nanpercentile(AP, 75)]
       a = [str(x) for x in a]

       # Write statistics to file
       a_file.write("\t".join(a)+"\n")
       print(i)
   # Close output files
   a_file.close()
   b_file.close()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
