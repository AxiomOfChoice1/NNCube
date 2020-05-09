# I'll be using Hogg's interactive cube: https://github.com/davidwhogg/MagicCube
# This will just be to show the moves. It's in the MagicCube folder
# I'll be using pycuber (not installed) to represent the cube and feed into NN

import sys
sys.path.insert(0, './/MagicCube//code//')
from cube_interactive import *
# This has built in rotate
import pycuber as pc

import A_Neural_Network as ann
from matplotlib import widgets
import matplotlib.pyplot as plt
import numpy as np


# Valid notational moves on a Rubik's cube
valid_moves = np.array(["R", "R'",
                        "U", "U'",
                        "L", "L'",
                        "F", "F'",
                        "B", "B'",
                        "D", "D'"], dtype='object')


# Generates a sample of size 'amount' with 'moves' random moves
# X[i] is i'th scrambled cube
# Y[i] is i'th next move
def generate_data(amount, moves):
    X = np.zeros((amount, 54))
    Y = np.zeros((amount, 12))
    for i in range(amount):
        algo = random_algorithm(moves) # random scramble of size 'moves'
        c = Cube_array()
        c.perform_algorithm(algo)
        X[i] = c.cube_to_array() # perform scramble and put to 1d array
        Y[i] = inverted_move_to_array(algo[-1]) # returns an array of last move inverted
                                # in other it's the next move to solve cube
    return X, Y


# Returns the next move to solve the current state of the scrambled cube
# Accepts a string of the move ex: "R'"
def inverted_move_to_array(move):
    array = np.zeros(12)
    m = move[0].upper()
    # This is the first letter of the move
    # What ever the move was, invert it
    # Example: if it was R', do R. If it was U do U'.
    if (len(move) == 2): # inverted move is clockwise
        if (m == "R"):
            array[0] = 1
        elif (m == "U"):
            array[1] = 1
        elif (m == "L"):
            array[2] = 1
        elif (m == "F"):
            array[3] = 1
        elif (m == "B"):
            array[4] = 1
        elif (m == "D"):
            array[5] = 1
    elif (len(move) == 1):
        if (m == "R"):
            array[6] = 1
        elif (m == "U"):
            array[7] = 1
        elif (m == "L"):
            array[8] = 1
        elif (m == "F"):
            array[9] = 1
        elif (m == "B"):
            array[10] = 1
        elif (m == "D"):
            array[11] = 1
    else:
        print("there's an error in inverted_move_to_array. move is : ", move)
        
    return array


def array_to_move(y):
    ind = np.argmax(y)
    
    if (ind == 0):
        return "R"
    if (ind == 1):
        return "U"
    if (ind == 2):
        return "L"
    if (ind == 3):
        return "F"
    if (ind == 4):
        return "B"
    if (ind == 5):
        return "D"
    if (ind == 6):
        return "R'"
    if (ind == 7):
        return "U'"
    if (ind == 8):
        return "L'"
    if (ind == 9):
        return "F'"
    if (ind == 10):
        return "B'"
    if (ind == 11):
        return "D'"
    print("There was an error in array_to_move: with vector y. Y has no chosen move")
    print("y is:", y) 
    return "R'" # Default move


# Generate a random 'letters' with amount of moves
# Calls check_no_redundancy to remove useless b2b moves like "R R'"
def random_algorithm(moves):
    algo = []
    for i in range(moves):
        algo.append(np.random.choice(valid_moves))
    return no_redundancy(algo)


# Checks that there are no repeated moves that cancel each other
# Example: R R' may be converted to R U
def no_redundancy(algo):
    n = len(algo)
    if (n == 1):
         # Case 1 one move : no redundancy just return it
        return algo
    for i in range(1, n):
        if (algo[i] == algo[i-1]):
            algo[i] = np.random.choice(valid_moves)
             # Case 2 redundancy, check again
            return no_redundancy(algo)
    # Case 3 no more redundancy
    return algo


# This is where pycuber gets converted to an array
class Cube_array():
    
    convert = {'orange' : 0.0,
               'yellow' : 0.2,
               'red' : 0.4,
               'green' : 0.6,
               'blue' : 0.8,
               'white' : 1.0}
    
    
    def __init__(self):
        self.pc_cube = pc.Cube()
        
    
    def perform_algorithm(self, algo):
        if (type(algo) == list):
            self.pc_cube.perform_algo(algo.copy())
        else:
            self.pc_cube.perform_algo(algo)
        
        
    # Returns a 1-dimensional representation of the cube
    def cube_to_array(self):
        a = [self.pc_cube.R, 
             self.pc_cube.U, 
             self.pc_cube.L, 
             self.pc_cube.F, 
             self.pc_cube.B, 
             self.pc_cube.D]
        array = np.zeros(54)
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    array[k + 3 * (j + 3 * i)] = self.convert[a[i][j][k].colour]
        return array


# Extended functionality of the interactive cube
class Cube(Cube):
    
    def __init__(self, NN):
        self.NN = NN
        self.pc_cube = Cube_array()
        # Constructor from Parent class
        super().__init__()
        
    
    # Override this method to have have control over figure
    def plot(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.int_cube = InteractiveCube(self)
        self.fig.add_axes(self.int_cube)
        self._add_move_widgets()
        return self.fig

    
    # Adds buttons, buttons are to rotate the cube
    def _add_move_widgets(self):
        Buttons = np.array([["D'", "D"], 
                            ["B'", "B"], 
                            ["F'", "F"], 
                            ["L'", "L"], 
                            ["U'", "U"], 
                            ["R'", "R"]])
        _ax_move = np.array(Buttons, dtype='object')
        self._btn_move = np.array(Buttons, dtype='object')
        x_pos = 0.84
        y_pos = 0.15
        
        for i in range(Buttons.shape[0]):
            for j in range(Buttons.shape[1]):
                x = x_pos + j*0.08
                y = y_pos + i*0.09
                
                # Reversed engineered their code (line 287) to add 'rotate' buttons
                _ax_move[i, j] = self.fig.add_axes([x, y, 0.07, 0.07])
                self._btn_move[i, j] = widgets.Button(_ax_move[i, j], Buttons[i, j])
                r = self._rotate_button(str(Buttons[i, j]), self.fig)
                self._btn_move[i, j].on_clicked(r)
                
        
        # Scramble button
        _ax_scramble = self.fig.add_axes([0.84, 0.69, 0.15, 0.07])
        self._btn_scramble = widgets.Button(_ax_scramble, "Scramble")
        s = self._scramble_button()
        self._btn_scramble.on_clicked(s)
        
        # Light Scramble button
        _ax_lscramble = self.fig.add_axes([0.84, 0.78, 0.15, 0.07])
        self._btn_lscramble = widgets.Button(_ax_lscramble, "Light Scramble")
        sl = self._lscramble_button()
        self._btn_lscramble.on_clicked(sl)
        
        # Predict move button from NN
        _ax_pred = self.fig.add_axes([0.84, 0.87, 0.15, 0.07])
        self._btn_pred = widgets.Button(_ax_pred, "Predict Move")
        p = self._predict_button()
        self._btn_pred.on_clicked(p)
        
        s = self._solve()
        self.int_cube._btn_solve.on_clicked(s)

    
    # Returns a function that enables the cube to be rotated
    def _rotate_button(self, notation, *args):
        if (len(notation) == 2):
            def f(*args):
                self.perform_algorithm([notation])
        else:
            def f(*args):
                self.perform_algorithm([notation])
        return f
    
    
    def _solve(self, *args):
        def s(*args):
             # just new cube, no need to unscramble (easier)
            self.pc_cube = Cube_array()   
        return s
    
    
    def _scramble_button(self, *args):
        def s(*args):
            r = random_algorithm(np.random.randint(15, 20))
            self.perform_algorithm(r)
        return s
    
    
    def _lscramble_button(self, *args):
        def sl(*args):
            r = random_algorithm(np.random.randint(2, 6))
            self.perform_algorithm(r)
        return sl
    
    
    def _predict_button(self, *args):
        def p(*args):
            y = self.NN.predict(self.pc_cube.cube_to_array())
            r = array_to_move(y)
            self.perform_algorithm([r])
        return p
    
    
    # Plain simple function to perform a rotation on the cube
    # Important: algorithm is an array of at least length 1
    def perform_algorithm(self, algorithm):        
        length = len(algorithm)
        for i in range(length):
            notation = algorithm[i]
            if (len(notation) == 2):
                self.int_cube.rotate_face(notation[0], -1)
            else:
                self.int_cube.rotate_face(notation)

        self.pc_cube.perform_algorithm(algorithm)



