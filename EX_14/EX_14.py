import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#%%
(Xtrain,Ytrain),(Xtest,Ytest)= tf.keras.datasets.mnist.load_data()

#plt.imshow(Xtest[0], cmap='Greys')


Xtrain_shaped=Xtrain.reshape(60000,28,28,1)/255
Xtest_shaped=Xtest.reshape(10000,28,28,1)/255

YtrainOH=np.array (pd.get_dummies(Ytrain))
YtestOH=np.array (pd.get_dummies(Ytest))

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(30,kernel_size=3,activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(30,kernel_size=3,activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax'),
     ])


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(Xtrain_shaped,YtrainOH,validation_data=(Xtest_shaped,YtestOH), 
          epochs=20,batch_size=100)

#%%
trainpredictions=np.argmax(model.predict(Xtrain_shaped),axis=1)
testpredictions=np.argmax(model.predict(Xtest_shaped),axis=1)
#%%

print('Accuracy in train data %.3f'% accuracy_score(Ytrain,trainpredictions))
print('Accuracy in test data %.3f'% accuracy_score(Ytest,testpredictions))

#%%
oldwrongpredictions=[104,115,247,321]



plt.figure()
plt.imshow(Xtest[104], cmap='Greys')
plt.figure()
plt.imshow(Xtest[115], cmap='Greys')
plt.figure()
plt.imshow(Xtest[247], cmap='Greys')
plt.figure()
plt.imshow(Xtest[321], cmap='Greys')

for n in oldwrongpredictions:
    prediction=np.argmax(model.predict(Xtest_shaped[n].reshape(1,28,28,1)))
    actual=Ytest[n]
    print('Index %.f prediction %.f actual %.f' %(n,prediction,actual))


'''wrongpredictions=[]

for i in range(0,len(Ytest)):
    if testpredictions[i] !=Ytest[i]:
       wrongpredictions.append(i) 
#%%     
for i in range(0,4):
           plt.figure()
           plt.imshow(Xtest[wrongpredictions[i]],cmap='Greys')'''

