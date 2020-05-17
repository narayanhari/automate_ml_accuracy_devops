from keras.datasets import mnist
from keras.optimizers import Adam

import json

from keras.models import Sequential,Model
from keras.layers import Dense,Input
from keras.utils.np_utils import to_categorical

from tweakModel import addLayer

dataset=mnist.load_data('mnist.db')
print(len(dataset))
train,test=dataset
print(len(train))

x_train,y_train=train
x_test,y_test=test

x_train1D=x_train.reshape(-1,28*28)
x_test1D=x_test.reshape(-1,28*28)
x_train = x_train1D.astype('float32')
x_test = x_test1D.astype('float32')

y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)

#inp_layer = Input(shape=(784,))
#out=addLayer(inp_layer,3,256)
json_file = open("parameters.json","r")
y=json.load(json_file)
num_layer=int(y['layer'])
num_units=int(y['units'])
num_epoch=int(y['epoch'])
inp_layer,out=addLayer(num_layer,num_units)
model = Model(inputs=[inp_layer],outputs=[out])
model.summary()

# model.add(Dense(units=264,activation='relu'))
# model.add(Dense(units=128,activation='relu'))
# model.add(Dense(units=32,activation='relu'))


#model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
h = model.fit(x_train, y_train_cat, epochs=num_epoch)
#
json_file.close()
model.save("NHM_mnist_ANN2.h5")
