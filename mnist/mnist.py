import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#http://pandas.pydata.org/pandas-docs/stable/10min.html
train_data = pd.read_csv('./../../../data/mnist/train.csv');
test_data = pd.read_csv('./../../../data/mnist/test.csv');

#https://www.kaggle.com/c/digit-recognizer/forums/t/4045/is-anyone-using-neural-networks/39515#post39515
y_train = train_data['label'].values
X_train = train_data.loc[:,'pixel0':].values

#http://rasbt.github.io/mlxtend/docs/data/mnist/
def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

plot_digit(X_train, y_train, 4)

def costFunction(params = 0, 
                 input_layer_size = 0,
                 hidden_layer_size = 0,
                 num_labels = 0,
                 X = 0, Y = 0, l = 0):
    J = 0
    grad = []
    return J, grad
    
    
print(costFunction())