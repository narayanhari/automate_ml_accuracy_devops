#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[46]:


dataset=mnist.load_data('mnist.db')


# In[47]:


print(len(dataset))


# In[48]:


train,test=dataset


# In[49]:


print(len(train))


# In[50]:


x_train,y_train=train
x_test,y_test=test


# In[7]:


# import cv2


# In[8]:


# cv2.imshow('hi',x_train[0])
# cv2.waitKey()
# cv2.destroyAllWindows()


# In[9]:


# y_train[0]


# In[51]:


# import matplotlib.pyplot as plt


# In[52]:


# plt.imshow(x_train[100])


# In[12]:


# y_train[100]


# In[13]:


# x_train.shape


# In[14]:


# x_test.shape


# In[15]:


x_train1D=x_train.reshape(-1,28*28)
x_test1D=x_test.reshape(-1,28*28)


# In[16]:


# x_train1D.shape


# In[17]:


x_train = x_train1D.astype('float32')
x_test = x_test1D.astype('float32')


# In[18]:


from keras.utils.np_utils import to_categorical


# In[19]:


y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)


# In[20]:


# y_train_cat


# In[21]:


# y_train_cat[0]


# In[22]:


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


# In[27]:


model.add(Dense(units=10,activation='softmax'))


# In[28]:


# model.summary()


# In[29]:


from keras.optimizers import Adam


# In[30]:


model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[31]:


h = model.fit(x_train, y_train_cat, epochs=2)


# In[32]:


# plt.plot(h.history['accuracy'])
# plt.plot(h.history['validation_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['training', 'validation'], loc='best')
# plt.show()


# In[44]:

model.save("NHM_mnist_ANN2.h5")

results = model.evaluate(x_test, y_test_cat, verbose=0)


# In[41]:


print(results)


# In[42]:


print(results[0])


# In[43]:


print(results[1])

file2 = open("test_accuracy.txt","w+")
x = results[1]
print(x)
file2.write(x)
file2.close()


# In[39]:


# y_pred=model.predict_classes(x_test)


# In[1]:


# y_pred


# In[41]:


# y_pred.shape


# In[42]:


# y_test


# In[51]:


# from sklearn.metrics import confusion_matrix


# In[43]:


# confusion_matrix(y_test,y_pred)


# In[44]:


# from sklearn.metrics import classification_report


# In[45]:


# print(classification_report(y_test,y_pred))


# In[46]:




# In[74]:


# x=x_test1D[0]
# # model.predict_classes(x)
# print(type(x))


# In[25]:


# from keras.models import load_model


# In[26]:


# m= load_model("NHM_mnist_ANN.h5")


# In[27]:


# y_reg=m.predict(x_test)


# In[29]:


# y_reg.shape


# In[ ]:




