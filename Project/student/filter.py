# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        F = np.matrix([[1, 0, 0, params.dt, 0, 0],
                      [0, 1, 0, 0, params.dt, 0],
                      [0, 0, 1, 0, 0, params.dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        return F
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        a = params.q/3*params.dt**3
        b = params.q/2*params.dt**2
        c = params.dt * params.q
        Q = np.matrix([[a, 0, 0, b, 0, 0],
                       [0, a, 0, 0, b, 0],
                       [0, 0, a, 0, 0, b],
                       [b, 0, 0, c, 0, 0],
                       [0, b, 0, 0, c, 0],
                       [0, 0, b, 0, 0, c]])
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        x = F * track.x
        P = F*track.P*F.transpose() + self.Q() 
        
        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H = meas.sensor.get_H(track.x)
        K = track.P * H.transpose() * np.linalg.inv(self.S(track, meas, H))
        x = track.x + K * self.gamma(track, meas)
        P = (np.identity(params.dim_state) - K * H) * track.P
        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        gamma = meas.z - meas.sensor.get_hx(track.x)

        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S = H * track.P * H.transpose() + meas.R
        return S
        
        ############
        # END student code
        ############ 