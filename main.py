import gzip
#import pickle
import math
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt
#import tensorflow as tf
#import input_data
import numpy as np


#The code to extract MNIST data is not included in this file as it would slow down the file. 
#d_import.py is added to the folder which contains the code. 
train_x=np.load('train_x.npy')
train_y=np.load('train_y.npy')
test_x=np.load('test_x.npy')
test_y=np.load('test_y.npy')


print "Logistic Regression Training"
print "Program Running..."
d=784   #Input Dimensions
k=10    #Classes
n=60000 #Total number
cntr=0
error=float("inf")
n_test=10000

x_bias=train_x
a=np.ones(n)
x_bias=np.vstack([a,np.transpose(x_bias)])

w=np.random.rand(d+1,k)
steps=1


for steps in range(0,1):
    error=np.zeros(n)
    ans=np.zeros((n,k))
    for n in range(0,n):
        aj=0
        ak=np.zeros(k)
        for x in range(0,k):
            ak[x]=np.dot(np.transpose(w[:,x]),x_bias[:,n])
            aj=aj+math.exp(ak[x])
            
        delta_E=np.zeros((k,d+1))
        y=np.zeros(k)
        E=0
        for x in range(0,k):
            y[x]=math.exp(ak[x])/aj
        
            #t vector
            t=np.zeros(k)
            t[int(train_y[n])]=1

            #delta E
            delta_E[x]=np.dot((y[x]-t[x]),x_bias[:,n])
              
            #cross entropy error
            E=E-(t[x]*math.log(y[x]))
        
        ans[n]=y
        error[n]=E
        delta_E=np.transpose(delta_E)
        #next w
        neta=0.01
        next_w=w-np.dot(neta,delta_E)
        w=next_w
  
#plt.plot(error)
#plt.show()

###################################test_lr##########################################
phi_test=test_x
a_test=np.ones(n_test)
phi_test=np.vstack([a_test,np.transpose(phi_test)])
ans_test=np.zeros((n_test,k))
for m in range(0,n_test):
    aj_test=0
    ak_test=np.zeros(k)
    for x in range(0,k):
        ak_test[x]=np.dot(np.transpose(w[:,x]),phi_test[:,m])
        aj_test=aj_test+math.exp(ak_test[x])
    y_test=np.zeros(k)
    for x in range(0,k):
        y_test[x]=math.exp(ak_test[x])/aj_test

    ans_test[m]=y_test

#right answers compare with true result
ans1_test=np.zeros((n_test,1))
ans2_test=np.zeros((n_test,1))
for x in range(0,n_test):
    for y in range(0,k):
        if(ans1_test[x][0]<ans_test[x][y]):
            ans1_test[x][0]=ans_test[x][y]
            ans2_test[x][0]=y
count_test =0
for x in range(0,n_test):
    if(test_y[x]==ans2_test[x][0]):
        count_test=count_test+1

print "Accuracy of Logistic Regression (MNIST test set of 10,000 images): "+str(count_test)+"/10000"
#################################usps_lr##########################################
test_usps_x=np.load('test_usps_x.npy')
test_usps_y=np.load('test_usps_y.npy')
n_test=19999
phi_test=test_usps_x
a_test=np.ones(n_test)
phi_test=np.vstack([a_test,np.transpose(phi_test)])
ans_test=np.zeros((n_test,k))
for m in range(0,n_test):
    aj_test=0
    ak_test=np.zeros(k)
    for x in range(0,k):
        ak_test[x]=np.dot(np.transpose(w[:,x]),phi_test[:,m])
        aj_test=aj_test+math.exp(ak_test[x])
    y_test=np.zeros(k)
    for x in range(0,k):
        y_test[x]=math.exp(ak_test[x])/aj_test

    ans_test[m]=y_test

#right answers compare with true result
ans1_test=np.zeros((n_test,1))
ans2_test=np.zeros((n_test,1))
for x in range(0,n_test):
    for y in range(0,k):
        if(ans1_test[x][0]<ans_test[x][y]):
            ans1_test[x][0]=ans_test[x][y]
            ans2_test[x][0]=y
count_test =0
for x in range(0,n_test):
    if(test_usps_y[x]==ans2_test[x][0]):
        count_test=count_test+1
print "Accuracy of Logistic Regression (USPS test set of 20,000 images): "+str(count_test)+"/20000"


####################################train_neural################################
print "Neural Network Training"
print "Program Running..."
m=784   #Input Dimensions
k=10
total=60000  
n=1000 #batch size
cntr=0
error=float("inf")
h=1024 #hidden nodes

w1=np.random.randn(m,h)
w2=np.random.randn(h,k)

a=np.ones((n,h))   ##bias
b=np.ones((n,k))

for a in range(0,6):
    for j in range(1,60):             #total = 6000 * batch size=60000
        x_bias=np.zeros((n,m))
        q=0
        for x in range((j-1)*n,(j)*n):
                x_bias[q]=train_x[x]        #x_bias is nothing but the train data
                q=q+1

        t=np.zeros((n,k))
        yk=np.zeros((n,k))
        z=np.zeros((n,h))
        sum_ak=np.zeros((n,1))
        E=0
    
        aj=np.dot(x_bias,w1) + a            #first activation function
    
        z=1/(1+np.exp(-1*aj))               #this is sigmoid function

    
        ak=np.dot(z,w2) + b                 #second activation function

        exp_ak=np.exp(ak)
        anand=np.sum(exp_ak,axis=1)
    
        for x in range(0,k):
            for y in range(0,n):
                yk[y][x]=exp_ak[y][x]/anand[y]               #output is calculated
    
   

        q=0
        for x in range((j-1)*n,(j)*n):
            t[q][int(train_y[x])]=1                     #t vector
            q=q+1
            
        Ek=yk-t                                         #delta 1
        Ej=np.multiply(np.multiply(z,(1-z)),np.dot(Ek,np.transpose(w2)))        #delta 2
    
        del_k=np.dot(np.transpose(x_bias),Ej)           #del 1   
        del_j=np.dot(np.transpose(z),Ek)                #del 2
    
        w1=w1-(0.001*del_k)
        w2=w2-(0.001*del_j)

##################################test_neural#####################################
m=784   #Input Dimensions
k=10
total=60000    #Classes
n=10000 #Total number
cntr=0
error=float("inf")
n_test=10000
h=1024 #hidden nodes

ans=np.zeros((10000,10))
for j in range(1,2):
    x_bias=np.zeros((n,m))
    q=0
    for x in range((j-1)*n,(j)*n):
            x_bias[q]=test_x[x]
            q=q+1
    a=np.ones((n,h))
    b=np.ones((n,k))
    
    
    
    t=np.zeros((n,k))
    yk=np.zeros((n,k))
    z=np.zeros((n,h))
    sum_ak=np.zeros((n,1))
    E=0
    
    aj=np.dot(x_bias,w1) + a
    
    z=1/(1+np.exp(-1*aj))

    
    ak=np.dot(z,w2) 

    exp_ak=np.exp(ak)
    anand=np.sum(exp_ak,axis=1)
    
    for x in range(0,k):
        for y in range(0,n):
           yk[y][x]=exp_ak[y][x]/anand[y]

    
ans1_test=np.zeros((n,1))
ans2_test=np.zeros((n,1))
for x in range(0,n):
    for y in range(0,k):
        if(ans1_test[x][0]<yk[x][y]):
            ans1_test[x][0]=yk[x][y]
            ans2_test[x][0]=y
count_test =0


for x in range(0,n):
    if(test_y[x]==ans2_test[x][0]):
        count_test=count_test+1
print "Accuracy of Single Neural Network (MNIST test set of 10,000 images): "+str(count_test)+"/10000"

#######################################usps_neural################################
m=784   #Input Dimensions
k=10
total=60000    #Classes
n=19999 #Total number
cntr=0
error=float("inf")
n_test=19999
h=1024 #hidden nodes

ans=np.zeros((n,m))
for j in range(1,2):
    x_bias=np.zeros((n,m))
    q=0
    for x in range((j-1)*n,(j)*n):
            x_bias[q]=test_usps_x[x]
            q=q+1
    a=np.ones((n,h))
    b=np.ones((n,k))

    
    t=np.zeros((n,k))
    yk=np.zeros((n,k))
    z=np.zeros((n,h))
    sum_ak=np.zeros((n,1))
    E=0
    
    aj=np.dot(x_bias,w1) + a
    
    z=1/(1+np.exp(-1*aj))

    
    ak=np.dot(z,w2) 

    exp_ak=np.exp(ak)
    anand=np.sum(exp_ak,axis=1)
    
    for x in range(0,k):
        for y in range(0,n):
           yk[y][x]=exp_ak[y][x]/anand[y]


ans1_test=np.zeros((n,1))
ans2_test=np.zeros((n,1))
for x in range(0,n):
    for y in range(0,k):
        if(ans1_test[x][0]<yk[x][y]):
            ans1_test[x][0]=yk[x][y]
            ans2_test[x][0]=y
count_test =0


for x in range(0,n):
    if(test_usps_y[x]==ans2_test[x][0]):
        count_test=count_test+1
print "Accuracy of Single Neural Network (USPS test set of 20,000 images): "+str(count_test)+"/20000"

                                          
#CNN code is commented as it would slow down the file. To operate just uncomment.
#It also has contains the CNN on USPS test set.
#####################################CNN####################################
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

 
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print "test accuracy %g"%accuracy.eval(feed_dict={x: test_usps_x, y_: test_usps_y, keep_prob: 1.0})
'''
