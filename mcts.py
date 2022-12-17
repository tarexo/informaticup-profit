import math
import tensorflow as tf
from copy import deepcopy
from helper.constants.settings import NUM_ACTIONS
import random as rand
from profit_gym import make_gym
from tqdm import tqdm
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

        self.dead_end = False  # Node has no legal actions

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
        # FIXME Is 'W' reward of action or given by the value network?
        # self.W = val

        action_probs = action_probs.numpy().flatten().tolist()  # tensor -> np array

        # FIXME Maybe mask out illegal game moves from action_probs and renormalize probabilities

        child_added = False
        solution_child = None

        for action in range(NUM_ACTIONS):
            env = copy_env(self.env)  # deepcopy(self.env)
            state, reward, done, legal, info = env.step(action)
            # env.render()
            if legal:
                child_added = True
                new_child = Node(
                    env=env,
                    game_state=state,
                    model=self.model,
                    parent=self,
                    action_taken=action,
                    prior_probability=action_probs[action],
                    final_state=done,
                    state_value=reward,
                )
                self.children.append(new_child)
            # if done:
            #     solution_child = new_child

        self.expanded = True
        if not child_added and not done:
            # ? How to handle dead ends? -> Remove for now
            no_solution_possible = self.remove_dead_end()

        return solution_child, child_added, no_solution_possible

    def remove_dead_end(self):
        no_solution_possible = False
        self.dead_end = True
        parent = self.parent
        parent.children = [child for child in parent.children if not child.dead_end]
        while len(parent.children) == 0:
            parent.dead_end = True
            parent = parent.parent
            if parent == None:
                print("There is no solution!")
                no_solution_possible = True
            parent.children = [child for child in parent.children if not child.dead_end]
        return no_solution_possible

    def select_child(self):
        # Select one of the available childs and return it
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
    # NOTE Idea is to train the network sucht that it behaves exactly like mcts
    def __init__(self, env, game_state, model):
        self.env = env
        self.game_state = game_state
        self.model = model
        return

    def run(self, num_runs=1600, is_train=True, tau=1.0):
        # TODO
        # Run the monte carlo search tree
        # ? What should be returned?
        # ? Does num_simluations refer to the depth of the tree or on node.simulate()?

        root = Node(self.env, self.game_state, self.model)
        # new_env = copy_env(self.env)
        self.env.render()
        # self.env.step(1)
        # self.env.render()
        # new_env.render()
        # new_env.step(10)
        # # new_env.step(11)
        # new_env.render()
        # self.env.render()

        for _ in tqdm(range(num_runs)):
            node = root
            # search_path = [node]

            # PHASE I: SELECT (a sequence of moves from the root to a leave)
            while node.expanded:
                node = node.select_child()

            # PHASE II: EXPAND (explore one more move)
            solution_node, added_child, no_solution_possible = node.expand()
            if no_solution_possible:
                raise Exception("No solution for MCTS possible!")

            if solution_node is not None:
                print("Found solution")
                node = solution_node
                while node is not None:
                    node.env.render()
                    node = node.parent
                solution_node.env.render()
                return

            # PHASE III: BACKUP (update all nodes on the path)
            if added_child:
                # Only backup if a step was taken
                self.backup(node)

        node.env.render()

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
        if node.parent is None:  # We are at the root
            return
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            return

        node = node.parent
        while node is not None:
            node.N += 1
            node.W += v  # FIXME v is a tensor
            node.Q = node.W / node.N
            node = node.parent


def copy_env(env):
    new_env = make_gym(
        width=env.unwrapped.width,
        height=env.unwrapped.height,
        copy=True,
        old_env=env,
    )
    new_env.reset(copy=True)
    return new_env
