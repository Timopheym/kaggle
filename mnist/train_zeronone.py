from scipy import optimize
from network import Network 

nt = Network()    
nn=nt.create([1, 1000, 1])

lamb=0.3
cost=1
alf = 0.2
xTrain = [[0], [1], [1.9], [2], [3], [3.31], [4], [4.7], [5], [5.1], [6], [7], [8], [9]]
yTrain = [[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]]

xTest= [[0.4], [1.51], [2.6], [3.23], [4.87], [5.78], [6.334], [7.667], [8.22], [9.1]]
yTest = [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]]

theta = nt.unroll(nn['theta'])
print(nt.runAll(nn, xTest))
theta =  optimize.fmin_cg(nt.costTotal, fprime=nt.backpropagation,
                x0=theta, args=(nn, xTrain, yTrain, lamb), maxiter=200)
print(nt.runAll(nn, xTest))