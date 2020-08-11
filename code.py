import numpy as np
import utils 
import math, random
import pandas as pd
import openpyxl

# Settings
csv_filename_train = "AI_DataTrain.csv"

Ques=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","Q11","Q12","Q13","Q14","Q15","Q16","Q17","Q18","Q19","Q20","Q21","Q22","Q23","Q24","Q25","Q26","Q27","Q28","Q29","Q30","Q31","Q32","Q33","Q34","Q35","Q36","Q37","Q38","Q39","Q40","Q41","Q42","Q43","Q44","Q45","Q46","Q47","Q48","Q49","Q50"]
l1=[]
l2=[]
alpha=0

class NN:

    def __init__(self, input_dim=None, output_dim=None, hidden_layers=None, seed=1):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given!")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # number of hidden nodes @ each layer
        self.network = self._build_network(seed=seed)

    # Train network
    def train(self, X, y, eta=0.5, n_epochs=200):
        for epoch in range(n_epochs):
            for (x_, y_) in zip(X, y):
                self._forward_pass(x_) # forward pass (update node["output"])
                yhot_ = self._one_hot_encoding(y_, self.output_dim) # one-hot target
                self._backward_pass(yhot_) # backward pass error (update node["delta"])
                self._update_weights(x_, eta) # update weights (update node["weight"])

    # Predict using argmax of logits
    def predict(self, X):
        ypred = np.array([np.argmax(self._forward_pass(x_)) for x_ in X], dtype=np.int)
        return ypred

    # ==============================
    #
    # Internal functions
    #
    # ==============================

    # Build fully-connected neural network (no bias terms)
    def _build_network(self, seed=1):
        random.seed(seed)

        # Create a single fully-connected layer
        def _layer(input_dim, output_dim):
            layer = []
            for i in range(output_dim):
                weights = [random.random() for _ in range(input_dim)] # sample N(0,1)
                node = {"weights": weights, # list of weights
                        "output": None, # scalar
                        "delta": None} # scalar
                layer.append(node)
            return layer

        # Stack layers (input -> hidden -> output)
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0]))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i]))
            network.append(_layer(self.hidden_layers[-1], self.output_dim))

        return network

    # Forward-pass (updates node['output'])
    def _forward_pass(self, x):
        transfer = self._sigmoid
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['output'] = transfer(self._dotprod(node['weights'], x_in))
                x_out.append(node['output'])
            x_in = x_out # set output as next input
        return x_in

    # Backward-pass (updates node['delta'], L2 loss is assumed)
    def _backward_pass(self, yhot):
        transfer_derivative = self._sigmoid_derivative # sig' = f(sig)
        n_layers = len(self.network)
        for i in reversed(range(n_layers)): # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = node['output'] - yhot[j]
                    node['delta'] = err * transfer_derivative(node['output'])
            else:
                # Weighted sum of deltas from upper layer
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = err * transfer_derivative(node['output'])

    # Update weights (updates node['weight'])
    def _update_weights(self, x, eta):
        for i, layer in enumerate(self.network):
            # Grab input values
            if i == 0: inputs = x
            else: inputs = [node_['output'] for node_ in self.network[i-1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    # dw = - learning_rate * (error * transfer') * input
                    node['weights'][j] += - eta * node['delta'] * input

    # Dot product
    def _dotprod(self, a, b):
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])

    # Sigmoid (activation function)
    def _sigmoid(self, x):
        return 1.0/(1.0+math.exp(-x))

    # Sigmoid derivative
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid*(1.0-sigmoid)

    # One-hot encoding
    def _one_hot_encoding(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=np.int)
        x[idx] = 1
        return x
           
for ipx in Ques:

    X, y, n_classes = utils.read_csv(csv_filename_train, target_name=ipx, normalize=True)
    N, d = X.shape

    hidden_layers = [5] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = 0.2 # learning rate
    n_epochs = 20 # number of training epochs
    n_folds = 5 # number of folds for cross-validation
    seed_crossval = 1 # seed for cross-validation
    seed_weights = 1 # seed for NN weight initialization
    print("\nStarting with",ipx)

    
    # Create cross-validation folds
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices

    # Train/evaluate the model on each fold
    acc_train, acc_test = list(), list()
    print("Cross-validating with {} folds...".format(len(idx_folds)))
    for i, idx_test in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_test)
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
              

        # Build neural network classifier model and train
        model = NN(input_dim=d, output_dim=n_classes,
                   hidden_layers=hidden_layers, seed=seed_weights)
        model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)

        # Make predictions for training and test data
        ypred_train = model.predict(X_train)
        ypred_test = model.predict(X_test)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==ypred_train)/len(y_train))
        acc_test.append(100*np.sum(y_test==ypred_test)/len(y_test))

        # Print cross-validation result
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_test = {:.2f}% ".format(
            i+1, n_folds, acc_train[-1], acc_test[-1], len(X_train), len(X_test)))

    # Print results
    print("  -> acc_train_avg = {:.2f}%, acc_test_avg = {:.2f}%".format(
        sum(acc_train)/float(len(acc_train)), sum(acc_test)/float(len(acc_test))))

    
    accuracy=sum(acc_test)/float(len(acc_test))
    l1.append(ipx)
    
    

    l2.append(accuracy)

    print("Question:",l1[alpha],"\nAccuracy:",l2[alpha])

    alpha+=1

print(l1,"\n",l2,"\n")
dicti = {'Question': l1, 'Accuracy': l2} 
df = pd.DataFrame(dicti)

# saving the dataframe 
df.to_excel('output.xlsx')
