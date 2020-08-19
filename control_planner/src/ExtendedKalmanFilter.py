#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class UNICAR(object):
    def __init__ (self, x_init=[0.0, 0.0, 0]):

        self.x_init = np.array(x_init, dtype=np.float32).reshape([-1,1])

        self.x_dim = 3
        self.u_dim = 2

        self.u_min = [0,-np.pi/6]
        self.u_max = [2, np.pi/6]

        self.pos_min = [-20.0, -20.0]
        self.pos_max = [20.0, 20.0]

        self.theta_min = -np.pi
        self.theta_max = np.pi

        self.dT = 0.05

        self.C = np.array([
                        [1,0,0],
                        [0,1,0],],dtype=float)

        self.w_sig = 0.2
        self.v_sig = np.array([0.1,0.1])
    
    def init_state(self):
        self.x = self.x_init

    def internal_step(self,u):
        u[0] = np.clip(u[0],self.u_min[0], self.u_max[0])
        u[1] = np.clip(u[1],self.u_min[1], self.u_max[1])

        dx0 = u[0] * np.cos(self.x[2,:])
        dx1 = u[0] * np.sin(self.x[2,:])
        dx2 = u[1] + self.w_sig * np.random.randn(1)

        dx_state = np.concatenate([dx0.reshape([-1,1]), dx1.reshape([-1,1]), dx2.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT * dx_state

        self.x[1,:] = np.clip(self.x[1,:], self.pos_min[1], self.pos_max[1])
        self.x[0,:] = np.clip(self.x[0,:], self.pos_min[0], self.pos_max[0])

        if self.x[2,:] < -np.pi:
            self.x[2,:] = self.x[2,:] + 2*np.pi

        elif self.x[2,:] > np.pi:
            self.x[2,:] = self.x[2,:] - 2*np.pi

        else:
            pass

    def external_step(self, u):
        self.internal_step(u)

        z = np.matmul(self.C, self.x) + np.reshape(self.v_sig * np.random.randn(2), (2,1))

        return z

class ExtendedKalmanFilter(object):
    def __init__(self, T=0.05):
        self.T = T

        self.X = np.array([0,0,0], dtype=float).reshape([-1,1])
        self.P = np.diag([2,2,2])

        self.Q = self.T * np.diag([0,0,0.04])
        self.R = np.diag([0.01,0.01])/self.T
        self.H = np.array([
                            [1,0,0],
                            [0,1,0],
        ], dtype=float)

    def prediction(self,u):
        dX_pre = np.zeros((3,1), dtype=float)

        dX_pre[0,:] = u[0] * np.cos(self.X[2,:])
        dX_pre[1,:] = u[0] * np.sin(self.X[2,:])
        dX_pre[2,:] = u[1]

        self.X += (self.T * dX_pre)

        if self.X[2,:] < -np.pi:
            self.X[2,:] = self.x[2,:] + 2*np.pi

        elif self.X[2,:] > np.pi:
            self.X[2,:] = self.x[2,:] - 2*np.pi

        else:
            pass
            
        self.calc_F(self.X,u)

        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def correction(self, Z):

        K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.R))

        Y = self.H.dot(self.X)

        self.X = self.X + K.dot(Z-Y)

        self.P = self.P - K.dot(self.H).dot(self.P)

    def calc_F(self, x, u):

        self.F = np.zeros_like(self.Q, dtype=float)

        self.F[0,2] = -u[0]*np.sin(x[2,:])
        self.F[1,2] = u[0] *np.cos(x[2,:])

        self.F = np.identity(x.shape[0]) + self.T * self.F

x_hat_hist, P_hist, z_hist, p_actual = [], [], [], []

env = UNICAR()
env.init_state()
ekf_estimator = ExtendedKalmanFilter()

u = 0.5 * np.ones((100,2))

for i in range(100):
    z = env.external_step(u[i,:])
    ekf_estimator.prediction(u[i,:])
    ekf_estimator.correction(z)

    print(ekf_estimator.X)

    x_hat_hist.append(ekf_estimator.X)
    P_hist.append(ekf_estimator.P)
    z_hist.append(z)
    p_actual.append(env.x[:2,:])

pos_hat = np.matmul(ekf_estimator.H, np.hstack(x_hat_hist))
z_np = np.hstack(z_hist)
p_np = np.hstack(p_actual)

plt.figure(figsize=(15,15))

plt.plot(pos_hat[0,:], pos_hat[1,:],"r-")
plt.plot(z_np[0,:], z_np[1,:], "k-")
plt.plot(p_np[0,:], p_np[1,:],"b-")

plt.ylabel("y")
plt.xlabel("x")

plt.show()