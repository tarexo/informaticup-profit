import math
import tensorflow as tf
from copy import deepcopy
from helper.constants.settings import NUM_ACTIONS
import random as rand


class Node:
    def __init__(
        self,
        env,
        game_state,
        model,
        parent=None,
        action_taken=None,
        prior_probability=None,
        final_state=False,
        state_value=0,
    ):
        self.env = env
        self.game_state = game_state
        self.model = model
        self.parent = parent
        self.children = []

        self.N = 0  # Number of times the node was visited
        self.W = state_value  # Raw value
        self.Q = 0  # W / N
        self.P = prior_probability

        self.expanded = False
        self.action_taken = action_taken
        self.final_state = final_state

    def expand(self):
        # TODO
        # Create all possible successor board states
        # Create Node objects from each state
        # Each new node gets self as parent
        # Add them as children to self
        # Return the Nodes as an array
        # ? Might need the model to expand?
        action_probs, val = self.make_prediction()

        # ? Alternatively take the reward given from the env as state value?
        self.W = val

        action_probs = action_probs.numpy().flatten().tolist()

        for action in range(NUM_ACTIONS):
            env = deepcopy(self.env)
            state, reward, done, legal, info = env.step(action)
            if legal:
                new_child = Node(
                    env=self.env,
                    game_state=state,
                    model=self.model,
                    parent=self,
                    action_taken=action,
                    prior_probability=action_probs[action],
                    final_state=done,
                    # state_value=reward,
                )
                self.children.append(new_child)
        self.expanded = True

    def select_child(self):
        # TODO
        # Select one of the available childs and return it
        # ? Selection may be based on the formula?
        children_visit_sum = 0
        for child in self.children:
            children_visit_sum += child.N

        c_puct = 1.0
        max_q_u = -float("inf")
        selected_child = None
        for child in self.children:
            Q = child.W / child.N if child.N > 0 else child.W
            U = c_puct * child.P * (math.sqrt(children_visit_sum) / 1 + child.N)
            if Q + U > max_q_u:
                max_q_u = Q + U
                selected_child = child
        return selected_child

    def make_prediction(self):
        game_state = tf.convert_to_tensor(self.game_state)
        game_state = tf.expand_dims(game_state, 0)
        action_probs, value = self.model(game_state)
        return action_probs, value


class MonteCarloTreeSearch:
    def __init__(self, env, game_state, model):
        self.env = env
        self.game_state = game_state
        self.model = model
        return

    def run(self, num_runs=100, is_train=True, tau=1.0):
        # TODO
        # Run the monte carlo search tree
        # ? What should be returned?
        # ? Does num_simluations refer to the depth of the tree or on node.simulate()?

        root = Node(self.env, self.game_state, self.model)

        for _ in range(num_runs):
            node = root
            # search_path = [node]

            # PHASE I: SELECT (a sequence of moves from the root to a leave)
            while node.expanded:
                node = node.select_child()

            # PHASE II: EXPAND (explore one more move)
            node.expand()

            # PHASE III: BACKUP (update all nodes on the path)
            self.backup()

        # PHASE IV: PLAY (final after repeating above ~1600 times)
        if not is_train:
            max_n = 0
            max_action = None
            for child in root.children:
                if child.N > max_n:
                    max_n = child.N
                    max_action = child.action_taken
            return max_action
        else:
            children_visit_sum = 0
            for child in root.children:
                children_visit_sum += child.N

            distribution = []
            actions = []
            for child in root.children:
                actions.append(child.action_taken)

                pi = child.N ** (1 / tau) / children_visit_sum
                distribution.append(pi)
            selected_action = rand.choices(
                population=actions, weights=distribution, k=1
            )
            return selected_action[0]

    def backup(self, node):
        v = node.W
        while node.parent is not None:
            parent = node.parent
            parent.N += 1
            parent.W += v
            parent.Q = parent.W / parent.N


def choose_child(node):
    # Repeated until leave node is reached
    # NOTE Pseudocode
    c_puct = 1.0  # Controls balance between exploration and exploitation
    best_U = -float("inf")
    chosen_action = None
    for action in node.actions:
        U = (
            c_puct
            * action.probability
            * (math.sqrt(sum_visits_over_children(node)) / 1 + node.get_child(action))
        )
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
