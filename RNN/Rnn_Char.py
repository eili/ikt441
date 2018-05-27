import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import SimpleRNN

data = [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1, 101)]]
target = np.array(target, dtype=float)

print(data)

data = data.reshape((1, 100, 1))
target = target.reshape((1, 100, 1))
print(data)

x_test = [[i for i in range(100, 200)]]
x_test = np.array(x_test, dtype=float)
y_test = [[i for i in range(101, 201)]]
y_test = np.array(y_test, dtype=float)

x_test = x_test.reshape((1, 100, 1))
y_test = y_test.reshape((1, 100, 1))

model = Sequential()
model.add(LSTM(1, input_shape=(100, 1), return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(data, target, epochs=2000, batch_size=1, verbose=1, validation_data=(x_test, y_test))

predict = model.predict(data)
