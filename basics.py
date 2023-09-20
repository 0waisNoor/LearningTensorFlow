import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

#initializing tensors
x = tf.constant(5, shape=(1,1),dtype=tf.float32)
x= tf.constant([[1,2,3],[4,5,6]])
x=tf.ones([3,3])
x=tf.eye(3) #identity matrix
x=tf.random.normal((3,3),mean=0,stddev=-1) #normal distribution
x=tf.random.uniform((1,3),minval=0,maxval=1)
x=tf.range(start=1,limit=10,delta=2)
x = tf.cast(x,dtype=tf.float64) #converting datatype


#mathematical operations
x = tf.constant([1,2,3])
y = tf.constant([3,4,5])

add = x+y
sub=x-y
mult=x*y
div=x/y
dotProduct = tf.tensordot(x,y,axes=1) #dot product summation


x = tf.constant([[1,2,3],[4,5,6]])
y = tf.constant([[7,8,9],[10,11,12],[13,14,15]])
z = x@y #this is the dot product representation


#indexing
ten = tf.constant([1,2,3,54,6])
print(ten[::-1])

#gathering  indices of matrix
x =tf.constant([1,2],
               [3,4],
               [5,6])

x_ind = tf.gather(x,tf.constant([2,4]))
print(x_ind)