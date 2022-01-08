# coding:utf-8

import random
import numpy as np

class QLearn:
    """
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    It use the best next choice of utility in later state to update the former state.
    """
    def __init__(self, num_states,actions, alpha, gamma, epsilon,min_exploration_rate = 0.01,n_episodes =1000,decay=True):
        self.q = np.zeros((num_states[0], num_states[1],len(actions)))
        
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions  # collection of choices
        self.epsilon = epsilon  # exploration constant
        self.min_eps = min_exploration_rate # epsilon min value
        self.exploration_decay_rate = (self.epsilon - self.min_eps)/n_episodes # epsilon decay rate
        self.decay = decay
        #print(self.q[0][2][0])

    # Get the utility of an action in certain state, default is 0.0.
    def get_utility(self, state, action):
        return self.q[state[0],state[1],action]          

    # When in certain state, find the best action while explore new grid by chance.
    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_utility(state, act) for act in self.actions]
            max_utility = max(q)
            # In case there're several state-action max values
            # we select a random one among them
            if q.count(max_utility) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_utility]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_utility)]
        return action

    # learn
    def learn(self, state1, action, state2, reward):
        old_utility = self.get_utility(state1, action)
        if old_utility == 0: # change to 0
          
            self.q[state1[0],state1[1], action] = reward
           

        # update utility
        else:
            next_max_utility = max([self.get_utility(state2, a) for a in self.actions])
            self.q[state1[0],state1[1], action] = old_utility + self.alpha * (reward + self.gamma * next_max_utility -old_utility)
            
        # Decay epsilon if valid
        if self.decay and self.epsilon > self.min_eps:
            self.epsilon -= self.exploration_decay_rate

        