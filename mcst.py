import math

class Node:
    def __init__(self, game_state, parent, children):
        self.game_state = game_state
        self.parent = parent
        self.children = children
        self.visit_count = 0
        self.value = 0
        self.prior = 0

    def expand():
        # TODO
        # Create all possible successor board states
        # Create Node objects from each state
        # Each new node gets self as parent
        # Add them as children to self
        # Return the Nodes as an array
        # ? Might need the model to expand?
        raise NotImplementedError

    def simulate():
        # TODO
        # Perform random actions on the board until
        #   Game ends (connection from deposit to factory created)
        #   Or no more actions are possible
        # ? What should be returned?
        raise NotImplementedError

    def select_child():
        # TODO
        # Select one of the available childs and return it
        # ? Selection may be based on the formula?
        raise NotImplementedError

class MCST:
    def __init(self, root):
        self.root = root
        self.search_path = None
        return

    def run():
        # TODO
        # Run the monte carlo search tree
        # ? What should be returned?
        # ? Does num_simluations refer to the depth of the tree or on node.simulate()?
        raise NotImplementedError

    

def choose_child(node):
    # Repeated until leave node is reached
    # NOTE Pseudocode
    c_puct = 1.0 # Controls balance between exploration and exploitation
    best_U = -float("inf")
    chosen_action = None
    for action in node.actions:
        U = c_puct * action.probability * (math.sqrt(sum_visits_over_children(node)) / 1 + node.get_child(action))
        if U > best_U:
            best_U = U
            chosen_action = action
    return chosen_action

def choose_action(node):
    # If serious play, choose best, else sample from distribution
    # NOTE Pseudocode
    # TODO Create probability distribution from values
    tau = 1.0
    best_pi = -float("inf")
    chosen_action = None
    for action in node.actions:
        pi = action.visits ** (1 / tau) / sum_visits_over_children(node)
        if pi > best_pi:
            best_pi = pi
            chosen_action = action
    return action

def sum_visits_over_children(node):
    sum = 0
    for child in node.children:
        sum += child.visits
    return sum