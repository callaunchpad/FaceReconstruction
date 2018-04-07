from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, Concatenate
def convBlock(numIn, numOut):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(numIn,)))
    model.add(Activation('relu'))
    model.add(Conv2D(numOut/2,kernel_size=(1,1),strides=(1,1),Activation='relu',input_shape=(numIn,numIn,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(numOut/2,(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(numOut,(1,1)))
    return model

def skipLayer(numIn, numOut):
   if numIn == numOut:
       return Activation('linear')
   else:
       seq = Sequential()
       seq.add(Conv2D(numOut,(1,1)))
       return seq

def Residual(num_in, num_out):
    model = Sequential()
    model1 = convBlock(num_in,num_out)
    model2 = skipLayer(num_in,num_out)
    model.add(Concatenate([model1, model2]))
    return model


model = Residual(10,10)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)