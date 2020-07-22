from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ctmc.ctmc import ctmc
from scipy.stats import norm
import copy

def obs(x):
    y = np.random.normal(loc = x,scale = 1)
    return y
def obs_lh(x,mu):
    y = obs(x)
    pyx = norm.pdf(y,loc = mu, scale = 1)
    return pyx

if __name__ == '__main__':
    T = 21
    dt = 0.005
    D = 3
    alpha = 1
    beta  = 1
    Q = np.random.gamma(shape =1.0,scale=1.0,size=(D,D))
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


    dt = 0.1
    p = mc.forward_expm(dt)

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
        np.fill_diagonal(b, 0)
        err[k] = np.linalg.norm(a-b)
    plt.figure(10)
    plt.plot(err)
    plt.show()


    times = np.array([0 ,T])
    sol = mc.backward_ode(times,p0)
    y = sol.y
    t = sol.t
    plt.figure(1)
    plt.plot(t, y[0, :])
    plt.plot(t, y[1, :])
    plt.plot(t, y[2, :])
    plt.show()


    times = np.array([0 ,T])
    sol = mc.forward_ode(times)
    y = sol.y
    t = sol.t
    plt.figure(2)
    plt.plot(t, y[0, :])
    plt.plot(t, y[1, :])
    plt.plot(t, y[2, :])
    plt.show()

   # times = np.array([0,3,5, 10,15,18])
   # z = np.random.gamma(shape =1.0,scale=1.0,size=(5,6))
    mc.rand_init()
    z = mc.sample()

    t_lh = np.array([0])
    t0 = 0
    lh = np.ones((mc.dims, len(z)))
    p = 0.8
    for i in range(0, len(z)):
        t0 = t0 + z[i][0]
        if np.random.uniform(low=0, high=1) < p:
            t_lh = np.concatenate((t_lh, t0 * np.ones((1))))
            for j in range(0, mc.dims):
                lh[j, i] = obs_lh(z[i][1], j) + 0.001

    (rho,t_rho) = mc.backward_ode_post(t_lh, lh)
    sol = mc.forward_ode_post(t_rho,rho)
    y = sol.y
    t_y = sol.t

    plt.figure(1)
    plt.plot(t_rho,rho[0,:])
    plt.plot(t_rho,rho[1,:])
    plt.figure(2)
    plt.plot(t_y, y[0, :])
    plt.plot(t_y, y[1, :])


    mc.reset_stats()
    mc.estimate_Q()
    a = copy.copy(mc.Q_estimate)

    M = 100
    err = np.zeros((M, 1))
    for k in range(0, M):
        p0 = np.ones((1, D)).flatten()
        p0[np.random.randint(low=0, high=D)] = 0
        p0 = p0 / sum(p0)
        mc.set_init(p0)
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
        print(mc.trans)
        print(mc.dwell)
        plt.figure(3)
        plt.plot(t_y,y[0,:])
        plt.show()
        a = copy.copy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        print(a)
        print(b)
        print(np.linalg.norm(a - b))
        err[k] = np.linalg.norm(a - b)

    plt.figure(11)
    plt.plot(err)
    plt.show()