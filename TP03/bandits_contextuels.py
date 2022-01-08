# -*- coding: utf-8 -*-
"""
Élève: Raissa Camelo
.. moduleauthor:: Valentin Emiya
"""

import numpy as np


class LinearBandits:
    """
    Linear bandit problem

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    n_features : int
        Number of features
    """

    def __init__(self, n_arms, n_features):
        self._theta = np.random.randn(n_arms,n_features) # The "true_values" for contextual bandits (theta etoile)

    @property
    def n_arms(self):
        return self._theta.shape[0]

    @property
    def n_features(self):
        return self._theta.shape[1]
    def get_theta(self):
        return self._theta

    def step(self, a, x):
        """
        Parameters
        ----------
        a : int
            Index of action/arm
        x : ndarray
            Context (1D array)

        Returns
        -------
        float
            Reward
        """

        assert 0 <= a
        assert a < self.n_arms # self._theta[:, a]
        return np.vdot(x, self._theta[a]) + np.random.randn() #dot product between x and action + random (0,1]
        # l'action est le theta , # la partie random est pour simuler un context real (probleme real)

    def get_context(self):
        """
        Returns
        -------
        ndarray
            Context (1D array)
        """
        return np.random.randn(self.n_features)

    def __str__(self):
        return '{}-arms linear bandit in dimension {}'.format(self.n_arms,
                                                              self.n_features)


class LinUCBAlgorithm():
    """
    

    Parameters
    ----------
    n_arms : int
        Number of arms
    n_features : int
        Number of features
    delta : float
        Confidence level in [0, 1]
    """
    

    def __init__(self, n_arms, n_features, delta = 0.05):
        # TODO à compléter
        self._n_arms = n_arms
        self._n_features = n_features
        
    
        self.A = []
        self.B = []
        for _ in range(n_arms):
            a = np.identity(n_features)
            b = np.zeros(n_features)
            self.A.append(a)
            self.B.append(b)
            
        self.alpha = 1 + np.sqrt((np.log((2/delta))/2))                                  
        
        # Initializer Ak
        # Crap I wrote in class when I wasnt understanding shit :)
        #context = self.get_context()
        #A = np.identity(self.n_features)
        #A =y + (context * context.transpose())
        #A = y + (context @ context.transpose())
        
        # Initializer Ck 
        
    

    @property
    def n_arms(self):
        #return self.A.shape[2]
        return len(self.A)

    @property
    def n_features(self):
        #return self.A.shape[0]
        return len(self.A[0])

    def get_action(self, x):
        """
        Choose an action

        Parameters
        ----------
        x : ndarray
            Context

        Returns
        -------
        int
            The chosen action
        """
        ucbs = [] 
        for k in range(self.n_arms):
            theta = np.dot(np.linalg.inv(self.A[k]),self.B[k])
            p1 = np.dot(x.transpose(), theta)
            p2 = self.alpha * np.sqrt(np.dot(np.dot(x.transpose(),np.linalg.inv(self.A[k])), x ))
            ucb = p1 + p2
            ucbs.append(ucb)
            
        return np.argmax(ucbs)
        

    def fit_step(self, action, reward, x):
        """
        Update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float
        x : ndarray

        """
        # m.a.j Ck
        #self.C = self.C + (x * r)
        #self.C = self.C + (x @ r)
        #self.C = self.C + (np.outer(x, r))
        # m.a.j Ak
        #self.A = self.A  (x * x.transpose())
        #self.A = self.A  (x @ x.transpose()) # (np.matmul = @)
        
        self.A[action] +=  (np.outer(x, x))
        self.B[action] +=  reward * x