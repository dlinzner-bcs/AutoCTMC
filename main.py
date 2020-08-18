from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ctmc.ctmc import ctmc
import copy
from environment.environment import emit_obs_M_norm


if __name__ == '__main__':
    """
    in this example, we:
    -generate a random ground truth ctms with random rate matrices
    -generate gaussian observations of unknown mean at random time points 
    -learn rate matrix and observation model in tandem
    -learns unsupervised assignemnt of latent-states to observations
    -TODO: implement non-parametric observation model to capture arbitrary data
    -TODO: catch if random rates are to slow (throws an error a.t.m.)
    """
#init ctmc
    #params of ctmc
    T = 21 #time window end
    dt = 0.0005 # timestep for simulation
    D = 2 # number of states of ctmc
    alpha = 0.1 # prior over num of transitions
    beta  = 0.1 # prior dwelling time

    #generate random rate matrix
    Q = np.random.gamma(shape =2.0,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    #generate random initial state
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)
    #prior assumption on observation model
    mu = np.random.uniform(low= -2,high = 2,size =(D,1))
    sig= np.ones((D))*0.2
    params = (mu,sig)

    #init ctmc
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)

#generate data
    # number of trajectories
    K   = 10
    # number of emissions per trajectory
    dat = emit_obs_M_norm(mc, 5, K)

#estimate rate matrix and obs model (only means atm)
#as the model learns assignemnt of latent-state -> observations unsupervised
#order of states in rate matrix and observation model are arbitrary
    M = 100 # number of EM iterations
    mc.reset_stats()
    mc.estimate_Q()
    for m in range(0, M):
        llh, sols = mc.process_emissions(dat)
        mc.update_obs_model(sols, dat)
        mc.estimate_Q()

        #log-likelihood
        print("log-likelihood:\n %s" % llh)
        #current obs params estimate
        print("mu_estimate:\n %s" % mc.params[0])
        print("mu_truth:\n %s" % np.array([0, 1]))

        mc.reset_stats()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        # mse of rate matrix estimate
        print("Q_estimate:\n %s" %a0)
        print("Q_truth:\n %s" %b)





