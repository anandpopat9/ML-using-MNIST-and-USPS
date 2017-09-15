import gzip
import pickle
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt
import numpy as np


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


train_y1=np.zeros(60000)
for x in range(0,50000):
    train_y1[x]=train_y[x]

for x in range(50000,60000):
    train_y1[x]=valid_y[x-50000]

train_x=np.vstack([train_x,valid_x])
#train_y=np.vstack([train_y,valid_y])
np.save('train_x',train_x)
np.save('train_y',train_y1)

np.save('test_x',test_x)
np.save('test_y',test_y)
#print type(train_x)
#print np.shape(train_x[0])
#plt.imshow(train_x[60000].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()