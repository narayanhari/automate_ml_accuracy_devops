from keras.models import load_model
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

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

m= load_model("NHM_mnist_ANN2.h5")

results = m.evaluate(x_test, y_test_cat, verbose=0)
print(results)
print(results[0])
print(results[1])

file2 = open("test_accuracy.txt","w+")
x = results[1]
print(x)

file2.write(str(x))
file2.close()