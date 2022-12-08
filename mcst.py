

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

    