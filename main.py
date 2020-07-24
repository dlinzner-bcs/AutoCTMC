from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ctmc.ctmc import ctmc
from scipy.stats import norm
from scipy.optimize import minimize
import copy

def obs(x):
    y = np.random.normal(loc = x,scale = 0.1)
    return y
def obs_lh(x,mu):
    y = obs(x)
    pyx = norm.pdf(y,loc = mu, scale = 0.1)
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


if __name__ == '__main__':

#init ctmc
    T = 21
    dt = 0.0005
    D = 2
    alpha = 0.1
    beta  = 0.1
    Q = np.random.gamma(shape =1,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)

    mu = np.arange(0,D)*1/2
    sig= np.ones((D))*0.1
    params = (mu,sig)
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)

#generate data

    M = 100
    K   = 20
    dat = gen_obs_M(mc, 20, K)

#estimate rate matrix and obs model (only means atm)

    mc.reset_stats()
    mc.estimate_Q()
    dat = emit_obs_M_norm(mc, 5, K)
    for m in range(0, M):
        llh, sols = mc.process_emissions(dat)
        mc.update_obs_model(sols, dat)

        #current obs params estimate
        print(mc.params[0])

        mc.estimate_Q()
        mc.reset_stats()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        # mse of rate matrix estimate
        print(np.linalg.norm(a0 - b))




