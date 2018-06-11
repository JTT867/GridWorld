import numpy as np
from collections import defaultdict


class QLearning:
    def __init__(self, actions):
        self.alpha = 0.01    # the learning rate
        self.gamma = 0.9    # the discount factor
        self.actions = actions
        self.epsilon = 0.1    # use epsilon for epsilon greedy
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def get_action(self, state):
        if np.random.uniform() > self.epsilon:
            # choose best action according to q-table
            action_values = self.q_table[state]
            argmax_actions = []    # the best actions may not be one, should randomly choose the next action
            for i in range(len(action_values)):
                if action_values[i] == np.max(action_values):
                    argmax_actions.append(i)
            next_action = np.random.choice(argmax_actions)
            # randomly choose an action from action base
        else:
            next_action = np.random.choice(self.actions)
        if self.epsilon > 0 :
            self.epsilon -= 0.00001    # reduce epsilon to stop the agent from exploring after several steps
            if self.epsilon < 0:
                self.epsilon = 0
        return next_action

    def learn(self, current_state, current_action, reward, next_state):
        next_action = np.argmax(self.q_table[next_state])
        # update q-table
        new_q = reward + self.gamma * self.q_table[next_state][next_action]
        self.q_table[current_state][current_action] = (1-self.alpha) * self.q_table[current_state][current_action] \
                                                      + self.alpha * new_q
        # current_q = self.q_table[current_state][current_action]
        # new_q = reward + self.gamma * max(self.q_table[next_state])
        # self.q_table[current_state][current_action] = (1-lr) * old_q + lr * new_q

