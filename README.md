# NNCube is a Python Program

This program is neural network that can solve a Rubik's Cube that's scrambled up to 5 moves.

The implementation was based off of a traditional neural network with the sigmoid function as the activation function, and backpropagation used for training. Additionally, for training, the genetic algorithm was implemented which usses accuracy as fitness.

When NNCube.py is run, there's an interactive console that has various options. When showing the cube, the MagicCube library was used to display an interactive cube on matplotlib, where you can scramble and have the neural network predict moves. Additionally, there's options to create other neural networks, see the loss & accuracy, and training options.

MagicCube was used to show an interactive cube on matplotlib. Buttons that were added to this interactive program were Predict Move, Light Scramble, Scramble, and Rubik's Cube notation: R R' U U' L L' F F' B B' D and D'.

The structure of the neural network is:
- 54 input nodes
- 12 output nodes

The hidden nodes are optional and making a nn with some is available. The input is a vector that represents the faces of the Rubik's Cube in the following order, Right, Up, Left, Front, Back, & Down, represented each by a number 1 - 5 and is normalized 0 - 1. The output is a vector that represents that move of either R, R', U, U', L, L', F, F', B, B', D, or D' in that order. So far it has above 50% accuracy.

**Libraries Used**
- copy
- MagicCube (installed in project)
- matplotlib
- numpy
- pickle
- pycuber
- sklearn for train_test_split
- sys

