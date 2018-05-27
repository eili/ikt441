from __future__ import print_function
from imagefilesloader import Imagefilesloader
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import random
import keras
batch_size = 128
num_classes = 2
epochs = 10

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Labeled Faces in the Wild
# Dataset from
# http://vis-www.cs.umass.edu/lfw/
# Each image is available as "lfw/name/name_xxxx.jpg"
# Each image is a 250x250 jpg

fr = Imagefilesloader()
trainingfolder = "./lfw_gender/"

(xdata, ydata) = fr.getgenderata(trainingfolder)
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.33, random_state=seed)

y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = x_train.reshape(x_train.shape[0],fr.sizex,fr.sizey,3)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.reshape(x_test.shape[0],fr.sizex,fr.sizey,3)


input_shape = (fr.sizex, fr.sizey, 3)

model = Sequential()

# Add a convolution layer:
model.add(Conv2D(32,
        kernel_size=(3,3),
        activation='relu',
        input_shape = input_shape))

model.add(Conv2D(64,
        kernel_size=(3,3),
        activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes,activation='sigmoid'))
model.summary()


model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.load_weights("./kerasweigth2.h5py", by_name=False)


model.fit(x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=batch_size, epochs=epochs,
            verbose=1)

#model.save_weights("./kerasweigth2.h5py", False)

score = model.evaluate(x_test,y_test,verbose=1)
print("Score", score)

score = model.evaluate(x_train,y_train,verbose=1)
print("Score", score)


