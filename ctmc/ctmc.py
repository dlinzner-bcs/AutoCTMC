import numpy as np
import scipy.linalg as spl
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.interpolate import interp1d
import copy

epsilon = 0.000001

class ctmc():
    """
    Continuous-Time Markov Chain.


    Parameters
    ----------
    Q   = intensity matrix
    p0  = inital state
    """
    def __init__(self, Q,p0,alpha,beta,T):
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

        self.alpha = alpha
        self.beta  = beta

        self.trans = np.zeros((self.dims,self.dims))
        self.dwell = np.zeros((self.dims,1))
        self.Q_estimate = np.zeros((self.dims,self.dims))

    def emit(self, func, **kwargs):
        func(self, **kwargs)  ## added self here
        return self

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
                        fac = (g(t)+epsilon)/(f(t)+epsilon)
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
        T = self.T

        t = np.concatenate((t, T*np.ones((1))))+1
        z = np.concatenate((z, np.ones((self.dims,1))), axis=1)
        rho0 = np.ones((1,self.dims)).flatten()
        rho  = np.ones((self.dims,1))
        times = np.ones((1,))
        for i in range(0,t.shape[0]-1):
           ti = np.array([T-t[i+1],T-t[i]])
           f = self.backward_ode(ti,rho0)
           assert f.status == 0
           rho0 = np.multiply(f.y[:,-1],z[:,i]).flatten()
           rho0 = rho0/sum(rho0)
           rho = np.concatenate((rho,f.y),axis = 1)
           times = np.concatenate((times, np.flip(f.t,axis=0)), axis=0)
        return (np.delete(rho,0,axis=1),np.delete(times,0,axis=0))

    def sample(self):
        T = self.T
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
           t = t + tau
           if t  <= T:
                z    = z    +   ((tau,s),)
        return z

    def statistics(self, z):
        T = np.zeros((self.dims,1))
        M = np.zeros((self.dims,self.dims))
        for i in range(0,len(z)):
            T[z[i][1]] = T[z[i][1]] + z[i][0]
            if i < len(z) - 1:
                M[z[i][1],z[i+1][1]] = M[z[i][1],z[i+1][1]] + 1
        return T,M

    def expected_statistics(self, y,t_y,rho,t_rho):
        eT = np.zeros((self.dims,1))
        eM = np.zeros((self.dims,self.dims))

        Q = self.Q

        def M_t(t,i,j):
            h = interp1d(t_y, y[i, :])
            if t <= np.max(t_rho):
                f = interp1d(t_rho, rho[i, :])
                g = interp1d(t_rho, rho[j, :])
                fac =  ((g(t) + epsilon) / (f(t) + epsilon))
            else:
                fac = 1
            Q_eff = Q[i,j] * fac*h(t)
            return Q_eff

        for i in range(0, self.dims):
            eT[i] = np.trapz(y[i,:],t_y)
            for j in range(0, self.dims):
                eM[i,j] = quad(M_t, 0, self.T,args=(i,j))[0]
        return eT,eM

    def update_statistics(self, z):
        T,M = self.statistics(z)
        self.trans = self.trans + M
        self.dwell = self.dwell + T
        return None

    def update_estatistics(self, y,t_y,rho,t_rho):
        eT,eM = self.expected_statistics(y,t_y,rho,t_rho)
        self.trans = self.trans + eM
        self.dwell = self.dwell + eT
        return None

    def estimate_Q(self):
        Q_estimate = np.zeros((self.dims,self.dims))
        for i in range(0,self.dims):
            for j in range(0, self.dims):
                Q_estimate[i,j] = (self.trans[i,j]+self.alpha)/(self.dwell[i]+self.beta)
        self.Q_estimate = Q_estimate
        return None



    def set_init(self,p0):
        self.p0 = p0
        return None

    def reset_stats(self):
        self.trans = self.trans*0
        self.dwell = self.dwell*0
        return None