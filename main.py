from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ctmc.ctmc import ctmc
from scipy.stats import norm


def obs(x):
    y = np.random.normal(loc = x,scale = 1)
    return y
def obs_lh(x,mu):
    y = obs(x)
    pyx = norm.pdf(y,loc = mu, scale = 1)
    return pyx

if __name__ == '__main__':
    T = 21
    D = 5
    Q = np.random.gamma(shape =1.0,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)
    mc = ctmc(Q,p0,T)

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


    z = mc.sample()

    times = np.array([0 ,T])
    sol = mc.backward_ode(times,p0)
    y = sol.y
    t = sol.t
    plt.figure(1)
    plt.plot(t, y[0, :])
    plt.plot(t, y[1, :])
    plt.plot(t, y[2, :])
    plt.plot(t, y[3, :])
    plt.show()


    times = np.array([0 ,T])
    sol = mc.forward_ode(times)
    y = sol.y
    t = sol.t
    plt.figure(2)
    plt.plot(t, y[0, :])
    plt.plot(t, y[1, :])
    plt.plot(t, y[2, :])
    plt.plot(t, y[3, :])
    plt.show()

   # times = np.array([0,3,5, 10,15,18])
   # z = np.random.gamma(shape =1.0,scale=1.0,size=(5,6))
    (y,t) = mc.backward_ode_post(t_lh, lh)

    plt.figure(3)
    plt.plot(t,y[0, :])
    plt.plot(t,y[1, :])
    plt.plot(t,y[2, :])
    plt.plot(t,y[3, :])
    plt.show()


    sol = mc.forward_ode_post(t,y)
    y = sol.y
    t = sol.t
    plt.figure(4)
    plt.plot(t,y[0, :])
    plt.plot(t,y[1, :])
    plt.plot(t,y[2, :])
    plt.plot(t,y[3, :])
    plt.plot(t, np.sum(y,axis=0))
    plt.show()