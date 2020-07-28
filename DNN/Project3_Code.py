import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import code

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras import optimizers

seed = 42
train = 'mnist_train.csv';
dataframe = pd.read_csv(train, header=0) ;
X = dataframe.iloc[:, 1:];
y = dataframe.iloc[:, 0];

def split(X, y):
    train_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)
    return X_train, X_test, y_train, y_test

binarizer = preprocessing.Binarizer()
X_binarized = binarizer.transform(X)
X_binarized = pd.DataFrame(X_binarized)
X_train, X_test, y_train, y_test = split(X_binarized, y)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

#Compile the model
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history1 = model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test), batch_size = 10000, epochs=10)
code.interact(local=locals())

print(history1.history.keys())
#  "Accuracy"
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train; lr = 0.01, bs =32', 'validation; lr = 0.01, bs =32','train; lr = 0.01, bs = 10000', 'validation; lr = 0.01, bs =10000','train; lr = 0.001, bs =32', 'validation; lr = 0.001, bs =32','train; lr = 0.001, bs = 10000', 'validation; lr = 0.001, bs =10000'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train; lr = 0.01, bs =32', 'validation; lr = 0.01, bs =32','train; lr = 0.01, bs = 10000', 'validation; lr = 0.01, bs =10000','train; lr = 0.001, bs =32', 'validation; lr = 0.001, bs =32','train; lr = 0.001, bs = 10000', 'validation; lr = 0.001, bs =10000'], loc='upper left')
plt.show()
test_error_rate = model.evaluate(x_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

model.save("trained_model_P3.h5")
test_data = pd.read_csv('mnist_test.csv')
Test_Data = test_data.iloc[:, 1:];
Test_Labels = test_data.iloc[:, 0];
# Reshaping the array to 4-dims so that it can work with the Keras API
Test_Data = Test_Data.values.reshape(Test_Data.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
Test_Data = Test_Data.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
Test_Data /= 255
print('Test_Data shape:', Test_Data.shape)
print('Number of images in ', Test_Data.shape[0])

model = keras.models.load_model("trained_model_P3.h5")
predictions = model.predict(x_test)
count = 0
Sub = np.subtract(predictions,a)
for i in range(0,10000):
    AA = np.count_nonzero(Sub[i,])
    if AA!= 0:
        count = count +1


Test_Accuracy = 1 - count/10000.
print(Test_Accuracy)
	