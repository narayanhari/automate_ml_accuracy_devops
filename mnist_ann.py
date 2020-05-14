#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[46]:


dataset=mnist.load_data('mnist.db')


# In[47]:


print(len(dataset))


train,test=dataset




print(len(train))



x_train,y_train=train
x_test,y_test=test


x_train1D=x_train.reshape(-1,28*28)
x_test1D=x_test.reshape(-1,28*28)


x_train = x_train1D.astype('float32')
x_test = x_test1D.astype('float32')

from keras.utils.np_utils import to_categorical


y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)


from keras.models import Sequential


# In[23]:


from keras.layers import Dense


# In[24]:


model=Sequential()


# In[25]:


model.add(Dense(units=512,input_dim=28*28,activation='relu'))
model.summary()


# In[26]:


model.add(Dense(units=264,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=32,activation='relu'))
# model.summary()


model.add(Dense(units=10,activation='softmax'))


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

h = model.fit(x_train, y_train_cat, epochs=2)
model.save("NHM_mnist_ANN2.h5")
