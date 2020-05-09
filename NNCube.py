import A_Neural_Network as ann
import A_Cube as ac
import matplotlib.pyplot as plt
import numpy as np
import pickle # To save the brains!
from sklearn.model_selection import train_test_split


# Referenced : https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
# To save and load objects (in my case the neural network)

# Referenced : https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# for the experimentation of hidden layers

# I'll be using Hogg's interactive cube: https://github.com/davidwhogg/MagicCube to show the cube
# This library is already install in this project: ./MagicCube
# I'll be using pycuber (not installed) to represent the cube and feed into NN

# (54 inputs, (nodes of hidden layers, separated by commas), 12 output nodes)
# Rubik's cube has 54 stickers : 6 faces * 9 stickers each
# Rubik's cube has 12 possible moves (side + dir): R R' U U' L L' F F' B B' D D'
# This neural network will predict the next move given a state of a Rubik's Cube


# Input key: {R:0, U:1, L:2, F:3, B:4, D:5} normalized, ie divided by 5
# Generates 500 samples with scramble of up to 5 moves
# X[i] is i'th scrambled cube with solution Y[i], but Y[i] only has next move
# X[i].shape : 54. organized: R U L F B D, top left sticker to bottom right sticker
# Y[i].shape : 12. organized: direction, and which side RULFB or D
# Example:
#     X[i] = [0.0, 0.2, 0.0, 0.4, 0.8, 0.8, 1.0, ...] : cube representation
#     Y[i] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, ..., 0.0] : move representation


def print_intro():
    print("\nNeural Network that tries to solve a Rubik's Cube")
    print("    1. Simulate the Rubik's Cube")
    print("    2. Train the Neural Network.")
    print("    3. Print Neural Network Info")
    print("    4. See the MSE and accuracy on a random sample")
    print("    5. Make a new neural network")
    print("    6. Load another neural network")
    print("    7. Rename the neural network")
    print("    8. Train with Genetic Algorithm")
    print("    9. 'Shake' the weights")
    print("    10. Save")
    print("    Any other number: Save & Quit")


# Assuming max_amount > 20
def gen_data(max_amount, max_moves): # Generate shuffled data
    one_move_amount = 30 # There are only 12 possible results for 1 move
    amount = round((max_amount-one_move_amount) / max_moves)
    X, Y = ac.generate_data(one_move_amount, 1)
    for moves in range(2, max_moves+1): # 2, 3, 4, ...
        X0, Y0 = ac.generate_data(amount, moves)
        X = np.concatenate((X, X0))
        Y = np.concatenate((Y, Y0))
    shuffle = np.random.permutation(len(X))
    X = X[shuffle] # Do the same shuffle on both arrays
    Y = Y[shuffle] 
    return X, Y


def auto_train(NN):
    
    try:
        learning_rate = float(input("\nEnter the learning rate: "))
    except ValueError:
        learning_rate = 0.1
        
    amount = 1000
    moves = 5
    X, Y = gen_data(amount, moves)
    
    start_accuracy = NN.accuracy(X, Y)
    start_loss = NN.loss(X, Y)
    
    iterations = 1000
    for i in range(iterations):
        if (i % (iterations // 11 + 1) == 0):
            print("\nProgress :", (i / iterations))
            print('Loss: ', NN.loss(X, Y))
            print('Accuracy: ', NN.accuracy(X, Y))  
        NN.fit(X, Y, learning_rate)
    
    print("\nStarted with loss: ", start_loss)
    print("Started with accuracy: ", start_accuracy)
    print("Trained the NN:", iterations, "times with generated sample size:", amount, "and max moves:", (moves+1))
    return NN


# saves objects as .pkl file
# not only for NN's, also for dataset
def save_nn(obj, name='brains'):
    filename = './/NeuralNetworks//' + name + '.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_nn(name):
    filename = './/NeuralNetworks//' + name + '.pkl'
    with open(filename, 'rb') as input:
        return pickle.load(input)





input_nodes = 54
output_nodes = 12

name = 'network'

try:
    NN = load_nn(name)
except FileNotFoundError:
    NN = ann.neural_network(input_nodes, () , output_nodes)
    save_nn(NN, name)
    
print("\nLoaded neural network with name: ", name)
NN.print_info()


while(True):
    
    print_intro()
    try:
        a = int(input("Please enter a number: "))
    except ValueError:
        a = 2570923840 # Retry
    
    
    if (a == 1):
        c = ac.Cube(NN) # Make cube and feed in Neural Network
        c.plot() # Plot it
        plt.show() # Show the plot : matplotlib.pyplot
        
        
    elif(a == 2):
        try:
            autotrain = int(input("\nPut in 0 to autotrain: "))
        except ValueError:
            autotrain = 0
        
        if (autotrain == 0):
            NN = auto_train(NN)
        else:
            print("\nWe'll generate some data")
            amount = int(input("    Enter the sample date size desired: "))
            moves = int(input("    Enter the amount of moves to shuffle cubes: "))
            iterations = int(input("    Enter the amount of times to train NN on this sample data: "))
            learning_rate = float(input("    Enter the learning rate: "))
            X, Y = ac.generate_data(amount, moves)
            
            start_accuracy = NN.accuracy(X, Y)
            for i in range(iterations):
                if (i % (iterations // 11 + 1) == 0):
                    print("\nProgress :", (i / iterations))
                    print('Loss: ', NN.loss(X, Y))
                    print('Accuracy: ', NN.accuracy(X, Y))  
                NN.fit(X, Y, learning_rate)
            print("\nStarted with accuracy: ", start_accuracy)
            print("Trained the NN:", iterations, "times with generated sample size:", amount, " and moves:", moves)
        
        
    elif(a == 3):
        NN.print_info()
        
        
    elif(a == 4):
        try:
            amount = int(input("\nEnter the amount for sample date size: "))
        except ValueError:
            amount = 100
        try:
            moves = int(input("    Enter the amount of moves to shuffle cubes: "))
        except ValueError:
            moves = np.random.randint(1, 10)
        X, Y = ac.generate_data(amount, moves)
        print('\nMSE: ', NN.loss(X, Y))
        print('Accuracy: ', NN.accuracy(X, Y), '\n')    
        
        
    elif(a == 5):
        save_nn(NN, name)
        print("\nNeural network with name:", name, " is saved!")
        chosen = False
        while (not chosen):
            try:
                length = int(input("\nEnter the amount of hidden nodes: "))
                chosen = True
            except ValueError:
                chosen = False
        t = ()
        for i in range(length):
            print("    Enter the number of nodes for layer", (i+1), ": ", end="")
            a = int(input())
            t = t + (a, )
        name = input("Give the brain a name: ")
        NN = ann.neural_network(input_nodes, t, output_nodes)
        print("Made neural network with name:", name)
        
        
    elif(a == 6):
        save_nn(NN, name)
        print("\nNeural network with name:", name, " is saved!")
        name = input("\nEnter the name of the neural network to load: ")
        NN = load_nn(name)
        print("Neural network with name:", name, "is loaded!")
    
    
    elif(a == 7):
        name = input("\nEnter the name desired: ")
        
        
    elif(a == 8):
        
        print("\nGenerating a data set")
        amount = 500 # Number for sample size to calculate fitness
        moves = 5  # random number of moves for cubes
        X, Y = gen_data(amount, moves)
        
        print("\nWe're gonna make some neural_networks. Please specify some specifics.")
        max_population = int(input("Enter the initial population: "))

        t = NN.hidden_layer_nodes # ()
        
        NNs = ann.ini_pop(max_population, input_nodes, t, output_nodes) # Initialize Population
        NNs[0] = NN
        ann.calculate_fitness(NNs, X, Y) # Calculate fitness
        
        print("\nCreated initial population!")
        generations = int(input("Enter the amount of generations wanted: "))
        
        for i in range(generations):
            if (i % (generations // 11 + 1) == 0):
                print("\nProgress :", (i / generations))
                print('Lowest loss: ', ann.lowest_loss(NNs).current_loss)
                print('Highest loss: ', ann.highest_loss(NNs).current_loss)
                print('The best accuracy: ', ann.best_accuracy(NNs).current_accuracy)
            selected = ann.selection(NNs) # Selection : Survival of the fittest
            crossed = ann.crossover(selected) # Crossover : Offspring
            crossed = ann.mutate(crossed) # Mutation : Small chance of randomness
            NNs = ann.join(selected, crossed, max_population)
            ann.calculate_fitness(NNs, X, Y) # Calculate fitness and do it again
        
        NN = ann.best_accuracy(NNs)
        print("\nThe best accuracy: ", NN.current_accuracy)
        print("The loss: ", NN.current_loss)
        print()
        save_nn(NN, name)
        print("\nNeural network with name:", name, " is saved!")
        
        
    elif (a == 9):
        
        
        try:
            percentage = float(input("\nEnter the percentage of weights to shake: 0.0 - 1.0: "))
        except ValueError:
            percentage = 0.1
        NN = ann.mutate([NN], 1, percentage)[0]
        print("Done!")
    
    
    elif (a == 10):
        save_nn(NN, name)
        print("\nNeural network with name:", name, " is saved!")
        
        
        
    # Finding optimal nn by testing hyperparameters
    elif (a == 123456789):
        print("\nGoing to look for an optimal neural network")

        try:
            X, Y = load_nn("sample_data_set")
        except FileNotFoundError:
            # Data set size is going to be ~ 4200 cubes
            # Population size for cubes scrambled up to 5 moves:
            # 12 + 12^2 + 12^3 + 12^4 + 12^5 = 271452
            # Takes around 10 seconds to generate
            print("\nGenerating a data set")
            X1, Y1 = ac.generate_data(25, 1)
            X2, Y2 = ac.generate_data(100, 2)
            X3, Y3 = ac.generate_data(1000, 3)
            X4, Y4 = ac.generate_data(1000, 4)
            X5, Y5 = ac.generate_data(2000, 5)
            X = np.concatenate((X1, X2, X3, X4, X5))
            Y = np.concatenate((Y1, Y2, Y3, Y4, Y5))
            shuffle = np.random.permutation(len(X))
            X = X[shuffle]
            Y = Y[shuffle]
            save_nn((X, Y), "sample_data_set")
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        # Validation is 10 % of the data
        X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size = 1/3)
        
        print("\nInput nodes is 54")
        print("Output nodes is 12")
        print("\nGenerating 1 hidden layer NNs up to 50 nodes in hidden layer")
        
        hidden_layer = 10
        validate = 100
        NNs = np.empty(hidden_layer, dtype='object')
        for i in range(hidden_layer):
            NNs[i] = ann.neural_network(input_nodes, ((i+5), ), output_nodes)
            for iterations in range(validate):
                # Train with validation data
                NNs[i].fit(X_validate, Y_validate, 0.1)
            # update the accuracy on test
            NNs[i].accuracy(X_test, Y_test)
        # Choose the best accuracy
        NN_1_hidden_layer = ann.best_accuracy_nof(NNs)
        
        print("Generating 2 hidden layer NNs up to 50 nodes in hidden layers")
        NNs = np.empty((hidden_layer, hidden_layer), dtype='object')
        for i in range(hidden_layer):
            for j in range(hidden_layer):
                NNs[i, j] = ann.neural_network(input_nodes, ((i+5), (j+5), ), output_nodes)
                for iterations in range(validate):
                    NNs[i, j].fit(X_validate, Y_validate, 0.1)
                NNs[i, j].accuracy(X_test, Y_test)
        NN_2_hidden_layer = ann.best_accuracy_nof(NNs)
        
        print("Generating 3 hidden layer NNs up to 50 nodes in hidden layers")
        NNs = np.empty((hidden_layer, hidden_layer, hidden_layer), dtype='object')
        for i in range(hidden_layer):
            for j in range(hidden_layer):
                for k in range(hidden_layer):
                    NNs[i, j, k] = ann.neural_network(input_nodes, ((i+5), (j+5), (k+5), ), output_nodes)
                    for iteration in range(validate):
                        NNs[i, j, k].fit(X_validate, Y_validate, 0.1)
                    NNs[i, j, k].accuracy(X_test, Y_test)
        
        NN_3_hidden_layer = ann.best_accuracy_nof(NNs)

        save_nn(NN_1_hidden_layer, "NN_1_hidden_layer")
        save_nn(NN_2_hidden_layer, "NN_2_hidden_layer")
        save_nn(NN_3_hidden_layer, "NN_3_hidden_layer")
        
        print("\nSaved the neural networks as: ")
        print("    NN_1_hidden_layer")
        print("    NN_2_hidden_layer")
        print("    NN_3_hidden_layer")
        
        
        
    # If user accidentally, or intentionally, put in a string
    elif (a == 2570923840):
        pass
    
    
    else:
        save_nn(NN, name)
        print("\nNeural network with name:", name, " is saved!")
        break




