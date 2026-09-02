import sys

class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None


if __name__ == '__main__':
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_or_prune = sys.argv[4]

    if (train_or_prune == "train"):
        with open("trained_tree.txt", "w") as f:
            print_tree(...)
    elif (train_or_prune == "prune"):
        prune_tree(...)
        with open("pruned_tree.txt", "w") as f:
            print_tree(...)