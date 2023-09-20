import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# flatten the image and convert to float 32 to save space. Also divide by 32 to normalize
x_train = x_train.reshape(-1,28*28).astype('float32')/255.0 
x_test = x_test.reshape(-1,28*28).astype('float32')/255.0 
print(x_train.shape)

'''
# Sequential API 
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)
])
'''

'''
# functional api
input =keras.Input(shape=28*28)
x = layers.Dense(512,activation='relu')(input)
x = layers.Dense(256,activation='relu')(x)
output = layers.Dense(10,activation="softmax")(x)
model = keras.Model(inputs=input,outputs=output)
'''


#uncomment this to print all all results of layers
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


#gets the 2nd last layer and gets its output, you cant directly input the last layer
model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output]) 

# model = keras.Model(input=model.inputs,outputs=[layer for layer in model.layers])


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), #parameter enables softmax, change this to true if using the sequential api else false
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

# the iterations in this model is total (trains records/32)*epoch
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=32)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)
print(model.summary())