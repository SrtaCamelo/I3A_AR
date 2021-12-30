# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import abc
import numpy as np


class BernoulliMultiArmedBandits:
    """
    Bandit problem with Bernoulli distributions

    Parameters
    ----------
    means : array-like
        True values (expectation of reward) for each arm
    """
    def __init__(self, means):
        self._means = np.array(means)
        assert np.all(0 <= self._means) #  Garantees elements are between 0 and 1
        assert np.all(self._means <= 1) # q*(a) = E[R|a]= pa , pa -> probability of arm a been 1 (chosen)

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        bool
            Reward obtained from playing arm `a` (true if win, false otherwise)
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.rand() < self._means[a] # reward is 'random' but respects the probability,  win or not  

    def __str__(self):
        return '{}-arms bandit problem with Bernoulli distributions'.format(
            self.n_arms)


class NormalMultiArmedBandits:
    """
    Bandit problem with normal distributions with unit variance.

    Parameters
    ----------
    means : array-like
        Mean values for each arm
    stds : array-like
        Standard deviation values for each arm (1 if None)
    """

    def __init__(self, means, stds=None):
        self._means = np.array(means)
        if stds is None:
            stds = np.ones_like(means)
        self._stds = stds

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        float
            Reward obtained from playing arm `a`
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.randn() * self._stds[a] + self._means[a] # value between 0-1 *std(a) + q*(a)

    def __str__(self):
        return '{}-arms bandit problem with Normal distributions'.format(
            self.n_arms)

    @staticmethod
    def create_random(n_arms):
        """

        Parameters
        ----------
        n_arms : int
            Number of arms or actions

        Returns
        -------

        """
        return NormalMultiArmedBandits(means=np.random.randn(n_arms))


class BanditAlgorithm(abc.ABC):
    """
    A generic abstract class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abc.abstractmethod
    def get_action(self):
        """
        Choose an action (abstract)

        Returns
        -------
        int
            The chosen action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair (abstract)

        Parameters
        ----------
        action : int
        reward : float

        """
        raise NotImplementedError


class RandomBanditAlgorithm(BanditAlgorithm):
    """
    A generic class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm Q(a)
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen N(a)
        self._n_estimates = np.zeros(n_arms)

    def get_action(self):
        """
        Choose an action at random uniformly among the available arms

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(self.n_arms)

    def fit_step(self, action, reward):
        """
        Do nothing since actions are chosen at random

        Parameters
        ----------
        action : int
        reward : float

        """
        pass


class GreedyBanditAlgorithm(BanditAlgorithm):
    """
    Greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms, dtype=int)

    def get_action(self):
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        
        i_nz = np.nonzero(self._n_estimates ==0)[0]
        if i_nz.size > 0:
            return i_nz[0]
        return np.argmax(self._value_estimates) # return the index of the max one
        

    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        self._n_estimates[action] += 1
        n = self._n_estimates[action]
        qa = self._value_estimates[action] 
        self._value_estimates[action] += (reward - qa)/n


class EpsilonGreedyBanditAlgorithm(GreedyBanditAlgorithm,
                                   RandomBanditAlgorithm):
    """
    Epsilon-greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Probability to choose an action at random
    """
    def __init__(self, n_arms=10, epsilon=0.1):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.epsilon = epsilon

    def get_action(self):
        """
        Get Epsilon-greedy action

        Choose an action at random with probability epsilon and a greedy
        action otherwise.

        Returns
        -------
        int
            The chosen action
        """
        p_rand = np.random.rand() < self.epsilon
        if p_rand:
            return np.random.randint(self.n_arms)
        else:
            #print("greedy")
            return GreedyBanditAlgorithm.get_action(self)

    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        GreedyBanditAlgorithm.fit_step(self, action, reward)

    
class UcbBanditAlgorithm(GreedyBanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    c : float
        Positive parameter to adjust exploration/explotation UCB criterion
    """
    def __init__(self, n_arms, c):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        
        # Estimation of the value of each arm Q(a) self._value_estimates 
        # Number of times each arm has been chosen N(a) self._n_estimates 

        self.c = c

    def get_action(self):
        """
        Get UCB action

        Returns
        -------
        int
            The chosen action
        """
        # Give a change to all bras be explored (when q(a) = 0)
        
        i_nz = np.nonzero(self._n_estimates ==0)[0]
        if i_nz.size > 0:
            return i_nz[0], [0,0,0,0]
        

        ucbs = self.get_upper_confidence_bound()
        i_arm = np.argmax(ucbs)
        return i_arm,ucbs # return the index of the max one + all ucbs for inference
            
        

    def get_upper_confidence_bound(self):
        ucbs = []
        t = np.sum(self._n_estimates)
        #print("t",t)
        for i in range(self.n_arms):
            exploitation = self._value_estimates[i]
            na = self._n_estimates[i]
            
            if na > 0:
                exploration = self.c * np.sqrt(np.log(t)/na)
            else:
                exploration = float('inf')
            
            ucb = exploitation + exploration
            ucbs.append(ucb)
            
            #print("na: ",na)
            #print("exploitation: ",exploitation)
            #print("exploration: ", exploration)
            #print("ucb",ucb, "\n")
        
        return ucbs


class ThompsonSamplingAlgorithm(BanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Alpha values for each arm
        self.alpha = np.ones(n_arms)
        # Beta values for each arm
        self.beta = np.ones(n_arms)
       

    def get_action(self):
        theta = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta[i] = np.random.beta(self.alpha[i], self.beta[i])
        #print(self.alpha, self.beta)
        return np.argmax(theta), self.alpha, self.beta
            

    def fit_step(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += 1 - reward
        #print(self.alpha)
        #print(self.beta)
        
        
