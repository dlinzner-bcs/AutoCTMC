import numpy as np
import scipy.linalg as spl
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import copy

class ctmc():
    """
    Continuous-Time Markov Chain.


    Parameters
    ----------
    Q   = intensity matrix
    p0  = inital state
    """
    def __init__(self, Q,p0,T):
        super().__init__()
        self.T    = T
        self.dims = p0.shape[0]
        try:
            assert Q.shape[0] == self.dims
            assert Q.shape[1] == self.dims
            for i in range(0, self.dims):
                 assert abs(sum(Q[i, :])) > 10^-8
        except AssertionError:
            raise ValueError('invalid adjacency matrix. try fix by setting consistent diagonals')
        self.Q    = Q
        try:
            assert sum(p0) == 1
        except AssertionError:
            raise ValueError('invalid initial state. try normalizing')
        self.p0   = p0



    def compute_propagator(self, dt):
        return spl.expm(self.Q * dt)

    def forward_expm(self, dt):
        U = self.compute_propagator(dt)
        p = np.dot(U, self.p0)
        return p

    def forward_ode(self, t):
        Q = self.Q
        def mastereq(t, x):
            dxdt =  np.dot(x,Q)
            return dxdt
        sol = solve_ivp(mastereq, t,self.p0,dense_output=True)
        return sol

    def forward_ode_post(self,t_rho,rho):
        Q = self.Q
        Q_eff = copy.copy(Q)
        def mastereq_t(t, x):
            for i in range(0,self.dims):
                for j in range(0, self.dims):
                    if t<=np.max(t_rho):
                        f = interp1d(t_rho, rho[i,:])
                        g = interp1d(t_rho, rho[j,:])
                        fac = (g(t)+0.01)/(f(t)+0.01)
                    else:
                        fac = 1
                    Q_eff[i,j] = Q[i,j]*fac
                Q_eff[i,i]=0
                Q_eff[i,i]=-sum(Q_eff[i,:])
            dxdt =  np.dot(x,Q_eff)
            return dxdt
        sol = solve_ivp(lambda t, y: mastereq_t(t,y), [0, self.T], self.p0, dense_output=True)
        return sol

    def backward_ode(self, t,rho0):
        Q = self.Q
        def mastereq_bwd(t, x):
            dxdt = np.dot(Q,x)
            return dxdt
        sol = solve_ivp(mastereq_bwd, t,rho0,dense_output=True)
        return sol

    def backward_ode_post(self, t,z):
        T = t[-1]
        rho0 = np.ones((1,self.dims)).flatten()
        rho  = np.ones((self.dims,1))
        times = np.ones((1,))
        for i in range(0,t.shape[0]-1):
           ti = np.array([T-t[i+1],T-t[i]])
           print(ti)
           f = self.backward_ode(ti,rho0)
           assert f.status == 0
           rho0 = np.multiply(f.y[:,-1],z[:,i]).flatten()
           rho0 = rho0/sum(rho0)
           rho = np.concatenate((rho,f.y),axis = 1)
           times = np.concatenate((times, np.flip(f.t,axis=0)), axis=0)
        return (np.delete(rho,0,axis=1),np.delete(times,0,axis=0))

    def sample(self, T):
        idx = np.arange(0,self.dims)
        Q   = self.Q
        t   = 0
        s   = np.random.multinomial(1,self.p0,size=None)
        s   = sum(s*idx)
        tau = np.random.exponential(-1/Q[s,s],size=None)
        z   = ((tau,s),)
        t   = t + tau
        while t <= T:
           p    = -Q[s,:]/Q[s,s]
           p[s] = 0
           s    = np.random.multinomial(1,p,size=None)
           s    = sum(s*idx)
           tau  = np.random.exponential(-1/Q[s, s])
           z    = z    +   ((tau,s),)
           t    = t    + tau
        return z

    def statistics(self, z):
        T = np.zeros((self.dims,1))
        M = np.zeros((self.dims,self.dims))
        for i in range(0,len(z)):
            T[z[i][1]] = T[z[i][1]] + z[i][0]
            if i < len(z) - 1:
                M[z[i][1],z[i+1][1]] = M[z[i][1],z[i+1][1]] + 1
        return T,M