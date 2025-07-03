import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#%%
(Xtrain,Ytrain),(Xtest,Ytest)= tf.keras.datasets.mnist.load_data()

#plt.imshow(Xtest[0], cmap='Greys')

Xtrain_flat=Xtrain.reshape(60000,784)/255
Xtest_flat=Xtest.reshape(10000,784)/255

YtrainOH=np.array (pd.get_dummies(Ytrain))
YtestOH=np.array (pd.get_dummies(Ytest))

model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation=tf.nn.relu,
                          input_shape=(Xtrain_flat.shape[1],)),

     tf.keras.layers.Dense(50,activation=tf.nn.relu),
     tf.keras.layers.Dense(30,activation=tf.nn.relu),
     tf.keras.layers.Dense(YtrainOH.shape[1],activation=tf.nn.softmax)
     ])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain_flat,YtrainOH,validation_data=(Xtest_flat,YtestOH), epochs=50,batch_size=100)

#%%
trainpredictions=np.argmax(model.predict(Xtrain_flat),axis=1)
testpredictions=np.argmax(model.predict(Xtest_flat),axis=1)
#%%

print('Accuracy in train data %.3f'% accuracy_score(Ytrain,trainpredictions))
print('Accuracy in test data %.3f'% accuracy_score(Ytest,testpredictions))

#%%

wrongpredictions=[]
for i in range(0,len(Ytest)):
    if testpredictions[i] !=Ytest[i]:
       wrongpredictions.append(i) 
#%%     
for i in range(0,4):
           plt.figure()
           plt.imshow(Xtest[wrongpredictions[i]],cmap='Greys')
plt.show()
