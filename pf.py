from numpy.random import uniform, randn, multivariate_normal, normal
import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
import scipy.stats
import matplotlib.pyplot as plt

''' 
SIR Particle Filter
'''

class PF:

    ''' Constructor '''
    def __init__(self, N=100, init_interval=[0.0,1.0],draw_particles_flag=False):
        self.N = N # number of particles 
        self.particles = np.zeros((self.N, 1))
        self.draw_particles_flag = draw_particles_flag
        self.particles[:,0] = uniform(init_interval[0],init_interval[1],size=self.N) # initialize particles
        self.weights = np.ones(self.N) / self.N # initialize weights
        self.estimate = None # current estimate
        self.estimate_var = None # current estimate variance
        self.estimation() # compute current estimate

    ''' compute PF estimate
        est_type: flag for MMSE or MAP estimate
    '''
    def estimation(self, est_type='MMSE'):
        # compute the expected state
        mean = np.average(self.particles, weights=self.weights, axis=0)
        # compute the variance of the state 
        self.estimate_var  = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        if est_type == 'MAP':
            # compute the MAP estimate
            argmax_weight = np.argmax(self.weights)
            self.estimate = self.particles[argmax_weight]
        else:
            # compute MMSE estimate
            self.estimate = mean

    ''' prediction step
        mu_omega: mean of the process model
        sigma_omega: variance of the process model
    '''
    def predict(self, mu_omega,sigma_omega):
        # move particles according to the process model
        omega = normal(mu_omega, sigma_omega*2,size=self.N)
        self.particles[:,0] += omega

    ''' update step
        z: 
    '''
    def update(self, z,std):
        # likelihood
        l = self.likelihood(z,std)
        self.weights *= l
        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    ''' sample subset of elements according to a list of indices
        indexes: list of indices that guides the elements sampling
    '''
    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    ''' SIS resampling '''
    def SIS_resampling(self):
        # resample if too few effective particles
        n_eff = 1. / np.sum(np.square(self.weights))
        if n_eff < self.N/2:
            indexes = systematic_resample(self.weights)
            self.resample_from_index(indexes)
            assert np.allclose(self.weights, 1/self.N)

    ''' RF likelihood
        z: 
    '''
    def likelihood(self, z,std):
        if not np.isnan(z): # in case of collected datum
            l = scipy.stats.norm.pdf(z,self.particles[:,0],std)
        else: # in case of non-collected datum
            l = 1 
        return l

    ''' PF plotting tool
    '''
    # def pf_plotting_tool(self, ax):
    #     if self.draw_particles_flag:
    #         self.plot_particles(ax)
    #     self.plot_estimate(ax)

    ''' draw particles
        ax: axis where to draw
    '''    
    # def plot_particles(self,ax):
    #     alpha = .20
    #     if self.N > 5000:
    #         alpha *= np.sqrt(5000)/np.sqrt(self.N)  
    #     rgba_colors = np.zeros((self.N,4))
    #     rgba_colors[:,1] = 1.0
    #     rgba_colors[:,3] = 10 + (self.weights**3)*100
    #     ax.scatter3D(self.particles[:, 0], self.particles[:, 1],0, c=rgba_colors[:,:-1], s=rgba_colors[:,-1], marker='o', alpha=0.1)

    ''' draw particle filter estimate
        ax: axis where to draw
    '''
    # def plot_estimate(self,ax):
    #     ax.scatter3D(self.estimate[0], self.estimate[1],0, color='g', marker='s')

        