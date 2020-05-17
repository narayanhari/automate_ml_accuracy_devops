from keras.models import Sequential,Model
from keras.layers import Dense,Input
from keras.utils.np_utils import to_categorical


def addLayer(a,b):
    a=int(a)
    b=int(b)
    inp_layer = Input(shape=(784,))
    q=Dense(units=b,activation='relu')(inp_layer)
    a=a-1
    b=int(b/2)
    for x in range(0,a):
        print("this is x in for loop",x)
        q=Dense(units=b,activation='relu')(q)
        b=int(b/2)
    out=Dense(units=10,activation='softmax')(q)
    return (inp_layer,out)