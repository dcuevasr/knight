#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:19:51 2017

@author: dario
"""

import knightClass as kc
import numpy as np

makn = kc.knight()
makn.set_goals(sensitivity = 100000000000)

#makn.alpha = 5

example = makn.full_inference(just_return = True)
makn.exampleFull()
example2 = makn.Example
#print(example['RealStates'][-1])



# for q in Q[V[:,t]==0]: print(q)
# for q in Q[V[:,t]==1]: print(q)