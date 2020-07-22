from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ctmc.ctmc import ctmc
from scipy.stats import norm
import copy

def obs(x):
    y = np.random.normal(loc = x,scale = 0.1)
    return y
def obs_lh(x,mu):
    y = obs(x)
    pyx = norm.pdf(y,loc = mu, scale = 0.1)
    return pyx

if __name__ == '__main__':
    T = 31
    dt = 0.0005
    D = 2
    alpha = 1
    beta  = 1
    Q = np.random.gamma(shape =.5,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)
    mc = ctmc(Q,p0,alpha,beta,T,dt)

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
    print(t_lh)
    print(lh)


    M = 100
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
        np.fill_diagonal(b, 0)
        err[k] = np.linalg.norm(a-b)
    plt.figure(10)
    plt.plot(err)
    plt.show()



    times = np.array([0,3,5, 10,15,18])
    z = np.random.gamma(shape =1.0,scale=1.0,size=(D,6))

    (rho,t_rho) = mc.backward_ode_post(times, z)
    sol = mc.forward_ode_post(t_rho,rho)
    y = sol.y
    t_y = sol.t

    f = np.divide(rho[0,:],rho[1,:])
    plt.figure(3)
    plt.plot(t_rho,f)
    plt.figure(4)
    plt.plot(t_y, y[0, :])
    plt.plot(t_y, y[1, :])
    plt.show()

    mc.reset_stats()
    mc.estimate_Q()
    a = copy.copy(mc.Q_estimate)

    M = 100
    err = np.zeros((M, 1))
    for k in range(0, M):
        mc.rand_init()
        z = mc.sample()

        t_lh = np.array([0])
        t0 = 0
        lh = np.ones((mc.dims, len(z)))
        p = 1
        for i in range(0, len(z)):
            t0 = t0 + z[i][0]
            if np.random.uniform(low=0, high=1) < p:
                t_lh = np.concatenate((t_lh, t0 * np.ones((1))))
                for j in range(0, mc.dims):
                    lh[j, i] = obs_lh(z[i][1], j)+0.001


        (rho, t_rho) = mc.backward_ode_post_marginal(t_lh, lh)
        sol = mc.forward_ode_post_marginal(t_rho, rho)
        y = sol.y
        t_y = sol.t

        mc.update_estatistics(y, t_y, rho, t_rho,a)
        mc.estimate_Q()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        print(np.linalg.norm(a0 - b))
        err[k] = np.linalg.norm(a0 - b)

    plt.figure(11)
    plt.plot(err)
    plt.show()