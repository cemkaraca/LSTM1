import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Setting up the ARM of order 1
n=10000   #total number of samples
a=0.8    #coefficient
u=np.zeros(int(3*n/2))   #array allocation
noise_std=0.2  #noise standard deviation
u[0]=0.          #value of the initial sample

for i in range(1,len(u)):
    eps=np.random.normal(loc=0,scale=1)   #random number from N(0,1)
    u[i]=a*u[i-1]+noise_std*eps
u=u[int(n/2):]  #discard the initial n/2 samples

#plt.plot(u)
#plt.show()
#print(len(u))
training_set = u[:7000]
test_set = u[7000:10000]

# normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1) #since working on a 1D array
scaled_train = scaler.fit_transform(training_set) #use fit_transform for training data?

test_set = test_set.reshape(-1,1)
scaled_test = scaler.transform(test_set) # use transform for the test data?

#print(scaled_test[:10])
print(scaled_train.shape)
print(scaled_test.shape)

#importing necessities for LSTM
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding

# Settşng up the LSTM architecture
n_steps =  10
n_features = 1
model_lstm = Sequential()
model_lstm.add(LSTM(64, activation="relu", input_shape=(n_steps, n_features)))
model_lstm.add(Dense(1))
# Compiling the model
model_lstm.compile(optimizer="Adam", loss="mse")
model_lstm.summary()

model_lstm.fit(scaled_train, scaled_test, epochs=20)

predicted = model_lstm.predict(scaled_test)
predicted = predicted.reshape(-1,1)
prediction = scaler.inverse_transform(predicted)


plt.plot(test_set, color="black", label="Real")
plt.plot(prediction, color="red", label="Predicted")
plt.show()
