import numpy as np
import scipy.linalg as spl
from scipy.integrate import solve_ivp
from scipy.stats import norm
from scipy.optimize import minimize
import copy

epsilon = 0.000001 #prevent numerical issues if likelihoods become to small

class ctmc():
    """
    Continuous-Time Markov Chain.


    Parameters
    ----------
    Q   = intensity matrix
    p0  = inital state
    alpha = hyperparam transitions
    beta  = hyper pram waiting time
    T     = Boundary time evolution
    dt    = time step
    params= parameters of observation model
    """
    def __init__(self, Q,p0,alpha,beta,T,dt,params):
        super().__init__()
        try:
            assert T>0
        except AssertionError:
            raise ValueError('invalid max time. time can not be negative.')
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
        self.dt = dt

        self.params = params

    def set_params(self,params):
        self.params = params
        return None

    def dat_lh(self,y,i):
        # observation model as likelihood of emission y under state i
        pyx = norm.pdf(y, loc=self.params[0][i], scale=self.params[1][i])
        return pyx

    def emit(self, func, **kwargs):
        func(self, **kwargs)  ## added self here
        return self

    def compute_propagator(self, dt):
        # compute propagator via matrix exponential
        return spl.expm(self.Q * dt)

    def forward_expm(self, dt):
        # solve forward ode with gt rate via matrix exponential
        U = self.compute_propagator(dt)
        p = np.dot(U, self.p0)
        return p

    def forward_ode(self):
        # solve forward ode with gt rate
        Q = self.Q
        def mastereq(t, x):
            dxdt =  np.dot(x,Q)
            return dxdt
        sol = solve_ivp(mastereq, [0,self.T],self.p0,dense_output=True)
        return sol

    def forward_ode_post(self,t_rho,rho):
        """
        solve forward ode with gt rate
        for bwd solution t_rho,rho
        """
        Q = self.Q
        Q_eff = copy.copy(Q)
        def mastereq_t(t, x):
            for i in range(0,self.dims):
                for j in range(0, self.dims):
                    if t<=np.max(t_rho):
                        result = np.where((t-self.dt <t_rho)*(t_rho<=t+self.dt))
                        f = rho[i, result[0][0]]
                        g = rho[j, result[0][0]]
                        fac = (g + epsilon) / (f + epsilon)
                    else:
                        fac = 1
                    Q_eff[i,j] = Q[i,j]*fac
                Q_eff[i,i]=0
                Q_eff[i,i]=-sum(Q_eff[i,:])
            dxdt = np.dot(x,Q_eff)
            return dxdt

        n_span = np.ceil(self.T / self.dt)
        np.ceil(n_span)
        t_span = np.linspace(0, self.T, int(n_span))
        sol = solve_ivp(lambda t, y: mastereq_t(t,y), [0, self.T], self.p0,t_eval=t_span, dense_output=True)
        return sol

    def forward_ode_post_marginal(self,t_rho,rho):
        """
        solve forward ode with current rate estimate
        for bwd solution t_rho,rho
        """
        Q = self.Q_estimate
        Q_eff = copy.copy(Q)
        p0 = np.ones((self.dims,1)).flatten()
        p0 = p0/sum(p0)

        def mastereq_t(t, x):
            for i in range(0,self.dims):
                for j in range(0, self.dims):
                    try:
                        result = np.where((t-self.dt <t_rho)*(t_rho<=t+self.dt))
                        f = rho[i, result[0][0]]
                        g = rho[j, result[0][0]]
                        fac = ((g + epsilon) / (f + epsilon))
                    except:
                        fac =1

                    Q_eff[i,j] = Q[i,j]*fac

                Q_eff[i,i]=0
                Q_eff[i,i]=-sum(Q_eff[i,:])
            dxdt =  np.dot(x,Q_eff)
            return dxdt

        n_span = np.ceil(self.T / self.dt)
        np.ceil(n_span)
        t_span = np.linspace(0, self.T, int(n_span))
        sol = solve_ivp(lambda t, y: mastereq_t(t,y), [0, self.T], y0 = p0,t_eval=t_span, dense_output=True)
        return sol

    def backward_ode(self, t,rho0):
        """
         solve backward ode with gt rate
         with one datum t,rho0
        """
        Q = self.Q
        def mastereq_bwd(t, x):
            dxdt = np.dot(Q,x)
            return dxdt

        n_span = np.ceil((t[1]-t[0])/self.dt)
        np.ceil(n_span)
        t_span = np.linspace(np.asscalar(t[0]), np.asscalar(t[1]), int(n_span))
        sol = solve_ivp(mastereq_bwd, t,rho0,t_eval=t_span,dense_output=True)
        return sol

    def backward_ode_marginal(self, t,rho0):
        """
         solve backward ode with current rate estimate
         with one datum t,rho0
        """
        Q = self.Q_estimate
        def mastereq_bwd(t, x):
            dxdt = np.dot(Q,x)
            return dxdt
        n_span = (np.ceil((t[1]-t[0])/self.dt))
        np.ceil(n_span)
        t_span = np.linspace(np.asscalar(t[0]),np.asscalar(t[1]),int(n_span))
        sol = solve_ivp(mastereq_bwd, t,rho0,t_eval=t_span,dense_output=True)
        return sol

    def backward_ode_post(self, t,z):
        """
         solve backward ode with gt rate
         for data t,z
        """
        T = self.T
        t = np.concatenate((t, T*np.ones((1))))
        z = np.concatenate((z, np.ones((self.dims,1))), axis=1)
        rho0 = np.ones((1,self.dims)).flatten()
        rho  = np.ones((self.dims,1))
        times = np.ones((1,))

        t_max = t.shape[0]-1

        for i in range(0,t_max+1):

           b = (t_max-i-1>=0)*(t[t_max-i-1])
           ti = np.array([T-t[t_max-i],T-b])

           f = self.backward_ode(ti,rho0)
           assert f.status == 0
           rho0 = np.multiply(f.y[:,-1],z[:,t_max-i-1]).flatten()
           rho0 = rho0/sum(rho0)
           rho = np.concatenate((rho,f.y),axis = 1)
           times = np.concatenate((times, f.t), axis=0)
        times  = np.delete(times,0,axis=0)
        rho    = np.flip(np.delete(rho,0,axis=1),axis=1)
        return (rho,times)

    def backward_ode_post_marginal(self, t,z):
        """
         solve backward ode with current rate estimate
         for data t,z
        """
        T = self.T
        t = np.concatenate((t, T * np.ones((1))))
        z = np.concatenate((z, np.ones((self.dims, 1))), axis=1)
        rho0 = np.ones((1, self.dims)).flatten()
        rho = np.ones((self.dims, 1))
        times = np.ones((1,))

        t_max = t.shape[0] - 1

        for i in range(0, t_max + 1):
            b = (t_max - i - 1 >= 0) * (t[t_max - i - 1])
            ti = np.array([T - t[t_max - i], T - b])
            f = self.backward_ode_marginal(ti, rho0)
            assert f.status == 0
            rho0 = np.multiply(f.y[:, -1], z[:, t_max - i - 1]).flatten()
            rho0 = rho0 / sum(rho0)
            rho = np.concatenate((rho, f.y), axis=1)
            times = np.concatenate((times, f.t), axis=0)
        times = np.delete(times, 0, axis=0)
        rho = np.flip(np.delete(rho, 0, axis=1), axis=1)
        return (rho, times)

    def sample(self):
        """
         sample path of ctmc
        """
        T = self.T
        idx = np.arange(0,self.dims)
        Q   = self.Q
        t   = 0
        s   = np.random.multinomial(1,self.p0,size=None)
        s   = sum(s*idx)
        tau = np.random.exponential(-1/Q[s,s],size=None)
        z   = ((tau,s),)
        t   = t + tau
        while t < T:
           p    = -Q[s,:]/Q[s,s]
           p[s] = 0
           p = p/sum(p)
           s    = np.random.multinomial(1,p,size=None)
           s    = sum(s*idx)
           tau = np.random.exponential(-1 / Q[s, s])
           t = t + tau
           if t  <= T:
                z    = z    +   ((tau,s),)
        return z

    def statistics(self, z):
        """
         compute statistics of sample path of ctmc
        """
        T = np.zeros((self.dims,1))
        M = np.zeros((self.dims,self.dims))
        for i in range(0,len(z)):
            T[z[i][1]] = T[z[i][1]] + z[i][0]
            if i < len(z) - 1:
                M[z[i][1],z[i+1][1]] = M[z[i][1],z[i+1][1]] + 1
        return T,M

    def expected_statistics(self, y,t_y,rho,t_rho,Q):
        """
         compute expected statistics of ctmc given posterior marginals
        """
        eT = np.zeros((self.dims,1))
        eM = np.zeros((self.dims,self.dims))

        for i in range(0, self.dims):
            eT[i] = np.trapz(y[i,:],t_y)
            for j in range(0, self.dims):
                l = min(len(y[i,:]),len(rho[i,:]))
                a = rho[j,0:l]
                b = rho[i,0:l]
                g = np.divide(a,b)
                f = np.multiply(y[i,0:l],g)
                eM[i,j] = Q[i,j]*np.trapz(f,t_y)
        return eT,eM

    def update_statistics(self, z):
        """
         update statistics
        """
        T,M = self.statistics(z)
        self.trans = self.trans + M
        self.dwell = self.dwell + T
        return None

    def update_estatistics(self, y,t_y,rho,t_rho,Q):
        """
         update expected statistics
        """
        eT,eM = self.expected_statistics(y,t_y,rho,t_rho,Q)
        self.trans = self.trans + eM
        self.dwell = self.dwell + eT
        return None

    def estimate_Q(self):
        """
         estimate rate
        """
        Q_estimate = np.zeros((self.dims,self.dims))
        for i in range(0,self.dims):
            for j in range(0, self.dims):
                a = self.trans[i,j]+self.alpha*(1+np.random.uniform(low=0, high=0,size=(1)).flatten())
                b = self.dwell[i]+self.beta*(1+np.random.uniform(low=0, high=0,size=(1)).flatten())
                Q_estimate[i,j] = a/b
            Q_estimate[i, i] = 0
            Q_estimate[i, i] = -sum(Q_estimate[i, :])
        self.Q_estimate = Q_estimate
        return None


    def set_init(self,p0):
        self.p0 = p0
        return None

    def rand_init(self):
        p0 = np.random.uniform(low=0, high=1,size=(1,self.dims)).flatten()
        p0 = p0 / sum(p0)
        self.set_init(p0)
        return None

    def reset_stats(self):
        self.trans = self.trans*0
        self.dwell = self.dwell*0
        return None

    def process_dat(self,t_lh, lh):
        (rho, t_rho) = self.backward_ode_post_marginal(t_lh, lh)
        sol = self.forward_ode_post_marginal(t_rho, rho)
        y = sol.y
        t_y = sol.t
        self.update_estatistics(y, t_y, rho, t_rho, self.Q_estimate)
        return None

    def process_emission(self, t_lh, emits):
        """
        process single emission-path
        input:
            t_lh = list of time points of emissions
            emits= list of emissions at time points
        output:
            llh = log-likelihood of model
            y   = posterior marginals
            t_y = time grid
            rho = backward solution
            t_rho = time grid
        """
        lh = np.ones((self.dims,len(t_lh)))
        for k in range(0,len(t_lh)):
            for j in range(0,self.dims):
                lh[j,k] = self.dat_lh(emits[k],j)+0.01

        (rho, t_rho) = self.backward_ode_post_marginal(t_lh, lh)
        sol = self.forward_ode_post_marginal(t_rho, rho)
        y = sol.y
        t_y = sol.t
        self.update_estatistics(y, t_y, rho, t_rho, self.Q_estimate)

        llh = self.llh() + self.llh_dat(t_lh,emits,y,t_y)

        return (llh,y,t_y,rho,t_rho)

    def process_emissions(self,dat):
        """
        process multiple emission-path
        input :
        dat = list of emission paths
        output:
        llh = log-likelihood of model of all paths
        sols= tuple of (posterior marginals, time grid)
        """
        K = len(dat)

        llh_full = 0
        sols = []
        for k in range(0, K):
            (llh, y, t_y, rho, t_rho) = self.process_emission(dat[k][0], dat[k][1])
            llh_full = llh_full +llh
            sols.append((y,t_y))
        return llh_full ,sols

    def llh(self):
        q = self.Q_estimate
        lnq = copy.copy(q)
        np.fill_diagonal(lnq,1)
        lnq = np.log(lnq)

        E_T = np.sum(np.multiply(self.dwell,q))
        E_M = np.sum(np.multiply(self.trans, lnq))

        llh = E_M-E_T
        return llh

    def llh_dat(self,t_lh,emits,y,ty):
        llh_dat = 0
        for k in range(0, len(t_lh)):
            t = t_lh[k]
            result = np.where((t - self.dt <ty) * (ty <= t + self.dt))
            for j in range(0, self.dims):
                llh_dat = llh_dat + np.log(self.dat_lh(emits[k], j))*y[j,result[0][0]]
        return llh_dat

    def llh_dat_full(self,sols,dat):
        M = len(sols)
        llh_dat = 0
        for m in range(0,M):
            t_lh = dat[m][0]
            emits= dat[m][1]
            t_y  = sols[m][1]
            y    = sols[m][0]
            for k in range(0, len(t_lh)):
                t = t_lh[k]
                result = np.where((t - self.dt <t_y) * (t_y <= t + self.dt))
                for j in range(0, self.dims):
                    llh_dat = llh_dat + np.log(self.dat_lh(emits[k], j))*y[j,result[0][0]]
        return llh_dat

    def update_obs_model(self,sols,dat):
        def mu_max(x):
            mc0 =copy.copy(self)
            params = (x,self.params[1])
            mc0.params = params
            a = np.min(abs(np.diff(np.array([x]))))
            llh = mc0.llh_dat_full(sols, dat) -10/(a+0.001)
            return -llh
        res = minimize(mu_max,x0 = self.params[0],method='Nelder-Mead', tol=1e-6)
        x = res.x
        params = (x, self.params[1])
        self.params = params
        return None




