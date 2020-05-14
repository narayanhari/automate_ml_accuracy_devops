from keras.models import load_model

# In[26]:


m= load_model("NHM_mnist_ANN2.h5")

results = m.evaluate(x_test, y_test_cat, verbose=0)


# In[41]:


print(results)


# In[42]:


print(results[0])


# In[43]:


print(results[1])

file2 = open("test_accuracy.txt","w+")
x = results[1]
print(x)
file2.write(str(x))
file2.close()