import numpy as np

'''
Kalman filter
'''

class Kalman_filter:

    ''' Constructor
        A: system matrix
        B: input matrix
        C: output matrix 
        Q: state variance
        R: measurement variance
        x0: initial state 
        P: (initial) prediction error variance
    '''
    def __init__(self,A,B,C,Q,R,x0,P):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0 # x is the state
        self.P = P

    ''' Prediction
    '''
    def predict(self):
        self.x = self.A*self.x
        self.P =  self.A*self.P*np.transpose(self.A) +  self.Q

    def update(self,z):
        y = z - self.C*self.x # innovation
        S = self.C*self.P*np.transpose(self.C) + self.R # innovation variance
        K = self.P*np.transpose(self.C)*S**(-1)  # Kalman gain
        self.x += K*y
        self.P = (1 - K*self.C) * self.P

