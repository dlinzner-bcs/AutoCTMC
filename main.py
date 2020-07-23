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
        print(t_lh)
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
        print(t_lh)
        z0 = (t_lh,y)
        z_full.append(z0)
    return z_full


if __name__ == '__main__':
    T = 21
    dt = 0.0005
    D = 2
    alpha = 0.1
    beta  = 0.1
    Q = np.random.gamma(shape =1,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    print(Q)
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)

    mu = np.arange(0,D)
    sig= np.ones((D))*0.1
    params = (mu,sig)
    print(params[0][1])

    mc = ctmc(Q,p0,alpha,beta,T,dt,params)

    z = mc.sample()
    t_lh = np.array([0])
    t0 = 0
    lh = np.ones((mc.dims,len(z)))
    p=0.5
    for i in range(0, len(z)):
        t0 = t0 + z[i][0]
        if np.random.uniform(low=0,high=1)<p:
          t_lh = np.concatenate((t_lh,t0*np.ones((1))))
          for j in range(0,mc.dims):
             lh[j,i] = obs_lh(z[i][1],j)

    M = 200
    err = np.zeros((M,1))
    for k in range(0,M):
        p0 = np.ones((1, D)).flatten()
        p0[np.random.randint(low=0,high=D)] = 0
        p0 = p0 / sum(p0)
        mc.set_init(p0)
        z = mc.sample()
        mc.update_statistics(z)
        mc.estimate_Q()
        a = copy.copy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        np.fill_diagonal(a,0)
        np.fill_diagonal(b,0)
        err[k] = np.linalg.norm(a-b)
    plt.figure(10)
    plt.plot(err)
    plt.show()

    sol = mc.forward_ode()
    y = sol.y
    t_y = sol.t
    plt.figure(2)
    plt.plot(t_y, y[0, :])
    plt.plot(t_y, y[1, :])

    mc.reset_stats()
    mc.estimate_Q()

    M = 100
    K   = 20
    dat = gen_obs_M(mc, 20, K)

    # for m in range(0,M):
    #     for k in range(0,K):
    #         mc.process_dat(dat[k][0],dat[k][1])
    #     print(mc.trans)
    #     print(mc.llh())
    #     mc.estimate_Q()
    #     mc.reset_stats()
    #
    #     a = copy.deepcopy(mc.Q_estimate)
    #     b = copy.copy(mc.Q)
    #     a0 = copy.deepcopy(mc.Q_estimate)
    #     np.fill_diagonal(a0, 0)
    #     np.fill_diagonal(b, 0)
    #     print(a0)
    #     print(b)
    #     print(np.linalg.norm(a0 - b))
    #     err[m] = np.linalg.norm(a0 - b)

    dat = emit_obs_M_norm(mc, 20, K)
    for m in range(0, M):
        for k in range(0,K):
            (llh,y,t_y,rho,t_rho) = mc.process_emissions(dat[k][0],dat[k][1])
        print(mc.trans)
        print(llh)
        mc.estimate_Q()
        mc.reset_stats()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        print(a0)
        print(b)
        print(np.linalg.norm(a0 - b))
        err[m] = np.linalg.norm(a0 - b)




