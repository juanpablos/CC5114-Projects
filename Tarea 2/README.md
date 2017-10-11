Tarea 2
===========================

Implementation of a recurrent neural network to predict some metrics in git respositories.
Comparison with a vanilla neural network to see which performs better.

To run the network, the following libraries are needed:

* Numpy
* TensorFlow (to run RNN_tensor_flow.py)
* Matplotlib
* Pandas (for data manipulation)

Files present:

* parser.py: used to convert the output files of the git commands to a csv file
* RNN_tensor_flow.py: the network used for the experiments
* RNN.py: an unfinished scratch implementation of a RNN using only numpy. Not finished.

For git metrics the following git commands were used:

* git log --name-status
* git log --shortstat (--stat also works but is a bigger file)