import numpy as np
import copy


# Referenced 8_NN_2layer.py (from classwork)

# Referenced : https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
# Referenced : https://github.com/nature-of-code/NOC-S18/blob/master/week10/neuroevolution-flappybird/ga.js
# For genetic algorithm goodies

select_percent = 0.10 # Select the top 10% for the next generation


def transfer(NN):
    NN_copy = NN.copy()
    new_NN = neural_network(NN.input_nodes, NN.hidden_layer_nodes, NN.output_nodes)
    new_NN.weights = NN_copy.weights
    return new_NN

def ini_pop(population, input_nodes, hidden_layer_nodes, output_nodes):
    NNs = np.empty(population, dtype='object')
    for i in range(population):
        NNs[i] = neural_network(input_nodes, hidden_layer_nodes, output_nodes)
    return NNs


def join(selected, crossed, population):
    NNs = np.empty(population, dtype='object')
    amount_selected = len(selected)
    NNs[0:amount_selected] = selected
    for i in range(amount_selected, population):
        NNs[i] = crossed[i-amount_selected]
    return NNs
        

# Base on accuracy instead of MSE
def calculate_fitness(NNs, X, Y):
    population = len(NNs)
    sum_correct = 0
    for i in range(population):
        NNs[i].loss(X, Y)
        sum_correct += NNs[i].correct(X, Y)
    for i in range(population):
        # The more accurate moves the better fitness score
        NNs[i].fitness_score = NNs[i].current_correct / sum_correct
    
    
def selection(NNs):
    initial_amount = len(NNs)
    select_amount = int(len(NNs) * select_percent) + 1
    selected = np.empty(select_amount, dtype='object')
    selected[0] = best_accuracy(NNs)
    for i in range(1, select_amount):
        count = 0
        while(True):
            count += 1
            r = np.random.rand() * (1 / (initial_amount * 6))
            index = np.random.randint(len(NNs))
            if (NNs[index].fitness_score > r):
                selected[i] = NNs[index].copy()
                NNs = np.delete(NNs, index)
                break
    return selected


def cross_rows(NN1, NN2):
    weights1 = NN1.weights
    weights_amount = len(weights1)
    NN = NN1.copy()
    for i in range(weights_amount):
        for j in range(weights1[i].shape[0]//2):
            NN.weights[i][2*j] = copy.deepcopy(NN2.weights[i][2*j])
    return NN


def cross_cols(NN1, NN2):
    weights1 = NN1.weights
    weights_amount = len(weights1)
    NN = NN1.copy()
    for i in range(weights_amount):
        for j in range(weights1[i].shape[1]//2):
            NN.weights[i][:, 2*j] = copy.deepcopy(NN2.weights[i][:, 2*j])
    return NN


def cross_eles(NN1, NN2):
    weights1 = NN1.weights
    weights2 = NN2.weights
    weights_amount = len(weights1)
    NN = NN1.copy()
    for i in range(weights_amount):
        for j in range(weights1[i].shape[0]):
            for k in range(weights1[i].shape[1] // 2):
                NN.weights[i][j][2*k] = NN2.weights[i][j][2*k]
    return NN
    
    
    
def crossover(selected):
    select_amount = len(selected)
    cross_amount = int(select_amount * (1/select_percent)) + 1
    crossed = np.empty(cross_amount, dtype='object')
    for i in range(cross_amount):
        # A random brain
        NN1 = selected[np.random.randint(select_amount)]
        NN2 = selected[np.random.randint(select_amount)] 
        if (i % 3 == 0):
            # Cross with rows
            crossed[i] = cross_rows(NN1, NN2)
        elif (i % 3 == 1):
            # Cross with columns
            crossed[i] = cross_cols(NN1, NN2)
        else:
            # Cross by elements of the matrix
            crossed[i] = cross_eles(NN1, NN2)
    return crossed


# 10 % mutation rate, mutate 10 % of nodes per layer
def mutate(crossed, mutate_rate = 0.5, nodes_rate=0.1):
    cross_amount = len(crossed)
    
    for index in range(cross_amount):
        r = np.random.rand()
        if (r < mutate_rate):
            w = crossed[index].weights
            for i in range(len(w)):
                nodes_amount = w[i].size * nodes_rate
                a, b = w[i].shape
                while(nodes_amount > 0):
                    ind1 = np.random.randint(a)
                    ind2 = np.random.randint(b)
                    w[i][ind1][ind2] = np.random.rand() * 20 - 10
                    nodes_amount -= 1
    return crossed


def best_accuracy(NNs):
    m = len(NNs)
    NN = NNs[0]
    for i in range(1, m):
        if (NN.fitness_score < NNs[i].fitness_score):
            NN = NNs[i]
    return NN


# Best accuracy doesn't use fitness score as a basis
def best_accuracy_nof(NNs):
    NNs = NNs.flatten()
    m = len(NNs)
    NN = NNs[0]
    for i in range(1, m):
        if (NN.current_accuracy < NNs[i].current_accuracy):
            NN = NNs[i]
    return NN


def lowest_loss(NNs):
    m = len(NNs)
    NN = NNs[0]
    for i in range(1, m):
        if (NNs[i].current_loss < NN.current_loss):
            NN = NNs[i]
    return NN


def highest_loss(NNs):
    m = len(NNs)
    NN = NNs[0]
    for i in range(1, m):
        if (NNs[i].current_loss > NN.current_loss):
            NN = NNs[i]
    return NN


class neural_network():
    
    
    def __init__(self, input_nodes, hidden_layer_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.layers = len(hidden_layer_nodes) + 2
        self.hidden_layer_nodes = hidden_layer_nodes
        self.weights = self._set_weights(hidden_layer_nodes)
        self.current_correct = 0.0
        self.current_accuracy = 0.0
        self.current_loss = 1.0
        self.fitness_score = 0.0    
        
        
    def copy(self):
        NN = neural_network(self.input_nodes, self.hidden_layer_nodes, self.output_nodes)
        NN.weights = copy.deepcopy(self.weights) # Deep copy instead of refernce
        NN.curent_loss = self.current_loss
        NN.fitness_score = self.fitness_score
        return NN
        
        
    def print_info(self):
        print('\nNumber of input nodes : ', self.input_nodes)
        print('Number of hidden layers : ', self.layers - 2)
        print('Number of nodes for hidden layers : ', self.hidden_layer_nodes)
        print('Number of output nodes : ', self.output_nodes)
        print('Number of weight matrices : ', len(self.weights))
        for i in range(len(self.weights)):
            print('    weight matrix ', (i+1), ' has shape : ', self.weights[i].shape)
        print('')
        
    
    def print_weights(self):
        print('\n\nnumber of weights : ', len(self.weights))
        for i in range(len(self.weights)):
            print('\nweight matrix ', (i+1), ' has shape : ', self.weights[i].shape)
            print('\nweights : \n', self.weights[i])
    
    def fit(self, X, Y, learning_rate = 0.1):
        layers = self._feed_forward(X)
        self._back_prop(layers, Y, learning_rate)
    
    
    def predict(self, x):
        Y = x.copy()
        for i in range(len(self.weights)):
            Y = self._activation_function(Y@self.weights[i])
        return Y
    
    
    def correct(self, X, Y):
        amount = len(X)
        Y_pred = self.predict(X)
        
        sum_correct = 0
        for i in range(amount):
             # If the indeces are right, the move is right
            if (np.argmax(Y_pred[i]) == np.argmax(Y[i])):
                sum_correct += 1
            
        self.current_correct = sum_correct
        self.current_accuracy = sum_correct / len(Y)
        return self.current_correct
    
    
    def accuracy(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = 0
        for i in range(len(Y)):
             # The chosen highest move is the same
            if (np.argmax(Y_pred[i]) == np.argmax(Y[i])):
                accuracy += 1
        accuracy /= len(Y)
        self.current_accuracy = accuracy
        return accuracy
    
    
    def loss(self, X, Y):
        self.current_loss = np.mean(np.square(Y - self.predict(X)))
        return self.current_loss            
    
    
    def _activation_function(self, X):
         # Sigmoid function
        return 1 / (1 + np.exp(-X))

    
    def _activation_function_d(self, P):
        return P * (1 - P)
    
    
    def _back_prop(self, layers, Y, learning_rate = 0.1):
        n = len(layers)
        d_w = np.empty(n-1, dtype='object')
        

        Z = 2 * (Y - layers[-1]) * self._activation_function_d(layers[-1])    
        for i in range(n-2, -1, -1):
            for j in range(n-2, i, -1):
                if (d_w[i] is None):
                    d_w[i] = (Z @ self.weights[j].T) * self._activation_function_d(layers[j])
                else:
                    d_w[i] = (d_w[i] @ self.weights[j].T) * self._activation_function_d(layers[j])
            
            if (d_w[i] is None):
                d_w[i] = layers[i].T @ Z
            else:
                d_w[i] = layers[i].T @ d_w[i]
        
        # update the weights
        for i in range(n-1):
            self.weights[i] += learning_rate*d_w[i]
    
    
    # Includes input layer and output layer
    def _feed_forward(self, X):
        n = self.layers
        layers = np.empty(n, dtype='object')
        layers[0] = X.copy()
        for i in range(1, n):
            layers[i] = self._activation_function(layers[i-1]@self.weights[i-1])
        return layers
        
        
    def _set_weights(self, layers):
        length = len(layers)
        W = np.empty(length+1, dtype='object') # Initialize with number of hidden layers
        
        if (length == 0): # This means there are no hidden layers
            W[0] = np.random.rand(self.input_nodes, self.output_nodes) * 20 - 10
            return W
        
        W = np.empty(length+1, dtype='object') # Initialize with number of hidden layers
        W[0] = np.random.rand(self.input_nodes, layers[0]) # W_0
        W[-1] = np.random.rand(layers[-1], self.output_nodes) # W_(n-1)
        
        for i in range(1, length):
            t = (layers[i-1], layers[i])
            W[i] = np.random.rand(*t)
        return W
        


# a = [0, 1, 2, 3, 4]
# b = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


