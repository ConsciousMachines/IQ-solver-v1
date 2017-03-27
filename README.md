# IQ-solver-v1
several methods for solving IQ puzzles using machine learning 

Specifically this solves the Raven Progressive Matrices which seek to test logic ability in human IQ tests. 
Since the basis underlying the test is to check for the presence versus change of certain shapes in each row of the matrix, we use the classic Nerual Network solution that can learn any logic function of 2 inputs and 1 output: a 1-layer, 2 hidden unit network capable of solving XOR, and any other combination of 2 binary inputs to give any output (any permutation of truth table elements) for example:

I think for copy right reasons you have to find your own Raven IQ tests, but here is an example:

https://github.com/ConsciousMachines/IQ-solver-v1/blob/master/ex1.png

XOR
~~~~~~
X1 X2 Y
0 0 | 0
1 0 | 1
0 1 | 1
1 1 | 0

If you change the output Y to any other function, the neural network will be able to learn it, so it can learn any set / logic operation: union, intersection, difference, XOR, or a personalized function. 

Prior to this layer, we need to convert the matrix picture into an input the neural network can understand. We try 2 methods: 
1. V1: Convolutional network to squeeze the features of the image into a more compact representation
2. V2: Compress the image into 2 training sets (2 top rows) and 1 test set, the bottom row. Each top row is convolved, we filter out the difference in piexsl between each set (we don't care about the white pixels that don't change or show anything) and then the logic neural network operates by taking the pixels from the left image as X1, the middle image as X2, and the third image as Y. 



I plan to add more to this repository in the future with a seamless version of the network that reads the images without having to specify specific pixel locations, by using convolution and max pool to locate useful features in each Raven Matrix entry.
