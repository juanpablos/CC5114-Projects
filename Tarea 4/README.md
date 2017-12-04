# Genetic programming

A genetic programming example to find a set of operations that best aproximates to a certain target number or equation.

The AST representation is implemented in tree.py. The following functionalities are provided:
- Copy: a tree can be deep copied with the *copy* method, thus changes in the copied node dont change the original node. Usage is `tree.copy`.
- Printing: a tree can be printed by just creating a tree and using the *print* function -> `print(tree)`.
- Evaluation: a tree can be evaluated using the *eval* method, just create the tree and use `tree.eval()`.

To run the genetic programming algorithms run Main.py and set the parameters accordingly. In case you want to run an equation finding algorithm, you might have to modify fitness2 and set it as the chosen fitness function, along with adding variables to the terminal set.
