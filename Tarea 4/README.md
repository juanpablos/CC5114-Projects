# Genetic programming

A genetic programming example to find the set of operations that best aproximates to a certain target number.

The AST representation is implemented in tree.py. The following functionalities are provided:
- Copy: a tree can be deep copied with the *copy* method, thus changes in the copied node dont change the original node. Usage is `tree.copy`.
- Printing: a tree can be printed by just creating a tree and using the *print* function -> `print(tree)`.
- Evaluation: a tree can be evaluated using the *eval* method, just create the tree and use `tree.eval()`.

