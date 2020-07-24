"""
TODO: add dedicated environment class
"""

from __future__ import print_function
import numpy as np
from scipy.stats import norm
import copy

def obs(x):
    y = np.random.normal(loc = x,scale = 0.1)
    return y
def obs_lh(x,mu):
    y = obs(x)
    pyx = norm.pdf(y,loc = mu, scale = 0.2)
    return pyx
def gen_obs(mc,p,K):
    z_full = []
    for k in range(0,K):
        mc.rand_init()
        z = mc.sample()
        t0 = 0
        t_lh = np.zeros((len(z)))
        lh = np.ones((mc.dims, len(z)))
        for i in range(0, len(z)):
            t0 = t0 + z[i][0]
            t_lh[i] = t0
            if np.random.uniform(low=0, high=1) < p:
                for j in range(0, mc.dims):
                    lh[j, i] = obs_lh(z[i][1], j)+0.01

        z0 = (t_lh,lh)
        z_full.append(z0)
    return z_full
def gen_obs_M(mc,M,K):

    z_full = []
    for k in range(0,K):
        mc.rand_init()
        z = mc.sample()

        t0 = 0
        for i in range(0, len(z)):
            t0 = t0 + z[i][0]
        u = np.sort(np.random.uniform(low=0, high=t0, size=(M,1)).flatten())

        lh = np.ones((mc.dims, M))
        t_lh = np.zeros((M))
        n = 0

        t0 = 0
        for i in range(0, len(z)):
            tf = t0 + z[i][0]
            result = np.where((t0<u)*(u<=tf))
            for k in range(0,len(result[0])):
                for j in range(0, mc.dims):
                    lh[j, n] = obs_lh(z[i][1], j) + 0.01
                t_lh[n] = u[result[0][k]]

                n=n+1
            t0 = copy.copy(tf)

        z0 = (t_lh,lh)
        z_full.append(z0)
    return z_full
def emit_obs_M_norm(mc,M,K):

    z_full = []
    for k in range(0,K):
        mc.rand_init()
        z = mc.sample()

        t0 = 0
        for i in range(0, len(z)):
            t0 = t0 + z[i][0]
        u = np.sort(np.random.uniform(low=0, high=t0, size=(M,1)).flatten())

        y = np.ones((M))
        t_lh = np.zeros((M))
        n = 0

        t0 = 0
        for i in range(0, len(z)):
            tf = t0 + z[i][0]
            result = np.where((t0<u)*(u<=tf))
            for k in range(0,len(result[0])):
                for j in range(0, mc.dims):
                    y[n] =  obs(z[i][1])
                t_lh[n] = u[result[0][k]]

                n=n+1
            t0 = copy.copy(tf)
        z0 = (t_lh,y)
        z_full.append(z0)
    return z_full