from statistics import mode
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

"""download mnist data into train and test imgs
X_train -> (60000, 28, 28) [6000 stacks/imgs of 28x28 pixels]
y_train -> (6000,) [6000 rows, each corresponding to an img]
X_test -> (10000, 28, 28) [1000 stacks/imgs of 28x28 pixels]
y_test -> (1000,) [1000 rows, each corresponding to an img]
"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape data to fit model. (number_of_imgs, height, width, 1=greyscale)

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# one-hot encode target column. 1 for index that img maps
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# create model
model = Sequential()

# add model layers
"""
First 2 layers deal with input img => 2D
    1st : 64 nodes, 3x3 kernel, relu activation, takes input shape
    2nd: 32 nodes, 3x3 kernel, relu activation
Then flatten to connect to a dense layer
Dense layer used to create an output of size 10 (one for each 'object'), use softmax to give probability matrix in results. 
"""
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# compile model using accuracy to measure model performance
"""
adam adjusts learning rate throughout training
categorical_crossentropy loss function. lower score => model is performing better
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
"""
epochs: number of times model will cycle through data. more the better
accuracy will keep getting printed as you cycle through data
"""
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# predict first 4 images in the test set
print(model.predict(X_test[:4]))

# show actual results
print(y_test[:4])
