Convolutional Neural Networks assignment

The CNN Assignment should be Face recognition. The dataset is available from http://vis-www.cs.umass.edu/lfw/
Can you recognize male/female?
Can you recognize individualis? 

A subset of the data is grouped in two folders, male and female. 
Using the Keras library. Sets ⅓ of the data as test, rest is training.
Using Keras sequential model

Keras model:
Conv2D, 32, relu
Conv2d, 64, relu
MaxPool2D
Dropout
Flatten
Dense
Dropout
Dense

Results after 10 epochs
Males 1401, Females 923
Training:
Loss: 0.17 using binary crossentropy
Accuracy: 0.95

Test:
Loss: 0.44 using binary crossentropy
Accuracy: 0.81

