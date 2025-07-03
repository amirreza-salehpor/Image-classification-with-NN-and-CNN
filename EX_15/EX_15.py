import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from PIL import Image
import os


# Define folder path


target_size = (28, 28)  # Width, Height


#--------------------------------------------------------------------------
folder_path = "chest_xray/test/NORMAL"
X_Normal_test = []
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)  # Open image
        img_resized = img.resize(target_size)  # Resize image
        img_array = np.array(img_resized)  # Convert to NumPy array
        if img_array.ndim == 2:
            X_Normal_test.append(img_array)



Y_Normal_test = np.ones(len(X_Normal_test))  # Creates an array with 5 ones
#--------------------------------------------------------------------------

# Define folder path
folder_path = "chest_xray/test/PNEUMONIA"

X_Pneumonia_test = []
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)  # Open image
        img_resized = img.resize(target_size)  # Resize image
        img_array = np.array(img_resized)  # Convert to NumPy array
        if img_array.ndim == 2:
            X_Pneumonia_test.append(img_array)
Y_Pneumonia_test = np.zeros(len(X_Pneumonia_test))  # Creates an array with 5 ones
#--------------------------------------------------------------------------
X_test_folder = np.concatenate((X_Normal_test, X_Pneumonia_test))
Y_test_folder = np.concatenate((Y_Normal_test, Y_Pneumonia_test))
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
folder_path = "chest_xray/train/NORMAL"
X_Normal_train = []
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)  # Open image
        img_resized = img.resize(target_size)  # Resize image
        img_array = np.array(img_resized)  # Convert to NumPy array
        if img_array.ndim == 2:
            X_Normal_train.append(img_array)
Y_Normal_train = np.ones(len(X_Normal_train))  # Creates an array with 5 ones
#--------------------------------------------------------------------------

# Define folder path
folder_path = "chest_xray/train/PNEUMONIA"

X_Pneumonia_train = []
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)  # Open image
        img_resized = img.resize(target_size)  # Resize image
        img_array = np.array(img_resized)  # Convert to NumPy array
        if img_array.ndim == 2:
            X_Pneumonia_train.append(img_array)
Y_Pneumonia_train = np.zeros(len(X_Pneumonia_train))  # Creates an array with 5 ones
#--------------------------------------------------------------------------

X_train_folder = np.concatenate((X_Normal_train, X_Pneumonia_train), axis=0)
Y_train_folder = np.concatenate((Y_Normal_train, Y_Pneumonia_train), axis=0)


X = np.concatenate((X_test_folder, X_train_folder), axis=0)
Y = np.concatenate((Y_test_folder, Y_train_folder), axis=0)

Xtrain, Xtest,Ytrain, Ytest=train_test_split(X, Y,test_size=0.3)

Xtrain = Xtrain.reshape(-1, 28, 28, 1)
Xtest = Xtest.reshape(-1, 28, 28, 1)

Ytrain = to_categorical(Ytrain, num_classes=2)
Ytest = to_categorical(Ytest, num_classes=2)

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(30,kernel_size=3,activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(30,kernel_size=3,activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax'),
     ])


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain,Ytrain,validation_data=(Xtest,Ytest), 
          epochs=50,batch_size=100)

#%%
trainpredictions=np.argmax(model.predict(Xtrain),axis=1)
testpredictions=np.argmax(model.predict(Xtest),axis=1)
#%%

Ytrain_single = np.argmax(Ytrain, axis=1)
Ytest_single = np.argmax(Ytest, axis=1)

# Accuracy calculation
print('Accuracy in train data %.3f' % accuracy_score(Ytrain_single, trainpredictions))
print('Accuracy in test data %.3f' % accuracy_score(Ytest_single, testpredictions))

#Xscaled= np.concatenate((X_Normal_test,Y_Normal_test), axis=0)
#Xtrain, Xtest,Ytrain, Ytest=train_test_split(X, Y,test_size=0.3)
'''plt.figure()
plt.imshow(X_Pneumonia_test[0], cmap='Greys')
plt.figure()
plt.imshow(result[234], cmap='Greys')
plt.show()'''