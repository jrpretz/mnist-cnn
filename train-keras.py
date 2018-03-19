import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from keras.optimizers import Adam

from keras.utils import np_utils

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator


np.random.seed(25)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print(model.summary())


generator = ImageDataGenerator(rotation_range=8, 
                               width_shift_range=0.08, 
                               shear_range=0.3,
                               height_shift_range=0.08, 
                               zoom_range=0.08)
test_gen = ImageDataGenerator()

train_generator = generator.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                    validation_data=test_generator, validation_steps=10000//64)


#model.fit(x=X_train,y=Y_train,batch_size=100,epochs=1)

model.save("trained-model-keras.hd5")

Y_test_pred = model.predict(x=X_test)
Y_train_pred = model.predict(x=X_train)
pred_test = np.argmax(Y_test_pred,axis=1)
pred_train = np.argmax(Y_train_pred,axis=1)

print(classification_report(pred_test,y_test))
print(classification_report(pred_train,y_train))

print(accuracy_score(pred_test,y_test))
print(accuracy_score(pred_train,y_train))
