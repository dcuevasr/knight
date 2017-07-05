#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:39:08 2017

@author: dario
"""

import numpy as np
import itertools
import actinfClass as af
from scipy.stats import norm


class knight(af.Actinf):
    def __init__(self, ObsNoise = 1, sensitivity = 1, rew1 = 1, rew2 = 2,
                 actNoise = 1, nT = 8+1):
        # Task parameters
        self.nS = 21
        self.nU = 2
        self.ObsNoise = ObsNoise
        self.sensitivity = sensitivity
        self.rew1 = rew1
        self.rew2 = rew2
        self.nT = nT
        self.actNoise = actNoise

        # Active inference parameters
        self.alpha = 64
        self.beta = 4
        self.lambd = 0.005 # or 1/4?
        self.gamma = 20

        # Get all matrices
        self._set_observation_matrix()
        self._set_ramp()
        self.set_transition_matrices()
        self.set_goals()
        self.set_initial_state()
        self.set_initial_belief()
        self.set_action_sequences()

        self.importMDP()

    def _set_observation_matrix(self):
        """ Defines the observation matrix."""
        A = np.zeros((self.nS, self.nS))
        aux_states = np.arange(21)
        for ix in range(self.nS): #cycle through columns
            A[:,ix] = norm.pdf(aux_states, ix, self.ObsNoise)
        A[-1,:] = np.zeros(self.nS)
        A[:,-1] = np.zeros(self.nS)
        A[-1,-1] = 1
        A[0,:] = np.zeros(self.nS)
        A[:,0] = np.zeros(self.nS)
        A[0,0] = 1
        self.A = A

    def _set_ramp(self):
        """ Sets the ramp function for step sizes."""
        self.R = np.hstack([2*np.ones(11), 2*np.arange(1,11)])

    def set_transition_matrices(self, actNoise = None, muRight = 1):
        """ Transition matrices based on self.R.
        Going left -> action 0
        Going right -> action 1
        """


        nU = self.nU
        nS = self.nS
        B = np.zeros((nU, nS, nS))

        if actNoise is None:
            actNoise = self.actNoise

        # Moving left
        aux_states = np.arange(nS)
        for col in range(nS):
            B[0,:,col] = norm.pdf(aux_states, col-self.R[col], actNoise)
            B[1,:,col] = norm.pdf(aux_states, col+muRight, actNoise)

        self.B = B

    def set_goals(self, sensitivity = None, rew1 = None, rew2 = None):
        """ Gets goals"""
        if sensitivity is None:
            sensitivity = self.sensitivity
        if rew1 is None:
            rew1 = self.rew1
        if rew2 is None:
            rew2 = self.rew2

        C = np.zeros(self.nS)
        C[0] = rew1
        C[-1] = rew2*sensitivity
        self.C = C/C.sum()
        self.lnC = np.log(self.C+np.exp(-160))

    def set_initial_state(self, iS = 10):
        """ Sets the initial state"""
        S = np.zeros(self.nS)
        S[iS] = 1
        self.S = S

    def set_initial_belief(self, iF = 10, through_obs = False):
        """ Sets initial belief. Must be executed after set_initial_state()"""
        D = self.S
        if through_obs is True:
            D = self.A.dot(D)
        self.D = D

    def set_action_sequences(self):
        """ Set matrix with all action sequences."""
        self.V = np.array(list(itertools.product([0,1], repeat=self.nT)))

