import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from network import Network 

#http://pandas.pydata.org/pandas-docs/stable/10min.html
train_data = pd.read_csv('./train.csv');
test_data = pd.read_csv('./test.csv');

#https://www.kaggle.com/c/digit-recognizer/forums/t/4045/is-anyone-using-neural-networks/39515#post39515
y_train = train_data['label'].values
X_train = train_data.loc[:,'pixel0':].values
X_test = test_data.loc[:,'pixel0':].values

y_train =list(map(lambda x: [x], y_train))

# y_train = y_train[:100]
# X_train = X_train[:100]

#http://rasbt.github.io/mlxtend/docs/data/mnist/
# def plot_digit(X, y, idx):
#     img = X[idx].reshape(28,28)
#     plt.imshow(img, cmap='Greys',  interpolation='nearest')
#     plt.title('true label: %d' % y[idx])
#     plt.show()

# plot_digit(X_train, y_train, 4)

nt = Network()    
nn=nt.create([784, 100, 1])

lamb=0.3
cost=1
alf = 0.005

i = 0                
results = []
while cost>0:
    cost=nt.costTotal(False, nn, X_train, y_train, lamb)
    delta=nt.backpropagation(False, nn, X_train, y_train, lamb)
    nn['theta']=[nn['theta'][i]-alf*delta[i] for i in range(0,len(nn['theta']))]
    i = i + 1
    print('Train cost ', cost[0,0], 'Iteration ', i)
    results = nt.runAll(nn, X_test)
    print(results)

np.savetxt("results.csv", results, delimiter=",")