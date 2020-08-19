#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class one_dim_env(object):
    def __init__(self, w_sig=0.5,v_sig=1,T=0.1):
        self.T = T
        self.A = np.array([[0,1],[0,0]])
        self.B = np.array([[0],[1]])

        self.Bw = np.array([[0],[1]])

        self.Ad = np.identity(2) + self.T * self.A
        self.Bd = self.T * self.B
        self.Bwd = self.T * self.Bw

        self.C = np.array([[1,0]])

        self.x = np.array([[0],[10]])

        self.w_sig = w_sig
        self.v_sig = v_sig

    def internal_step(self, u):
        self.x = np.matmul(self.Ad, self.x) + self.Bd * u + self.w_sig * np.random.randn(1)

    def external_step(self,u):
        self.internal_step(u)
        z = np.matmul(self.C, self.x) + self.v_sig * np.random.randn(1)
        return z

class KALMANFilter(object):
    def __init__(self, T=0.1, Q=np.diag([0,0.25]),R=1):
        self.T = T
        self.A = np.array([[0,1],[0,0]])
        self.B = np.array([[0],[1]])

        self.Ad = np.identity(2) + self.T * self.A
        self.Bd = self.T * self.B

        self.C = np.array([[1,0]])

        self.x_hat = np.array([[0],[0]])
        self.P = np.diag([10,10])

        self.Q = Q
        self.R = R/self.T

    def prediction(self, u):

        self.x_hat = np.matmul(self.Ad, self.x_hat) + self.Bd * u
        self.P = np.matmul(np.matmul(self.Ad, self.P), self.Ad.T) + self.Q

    def measurement(self, z):

        S = np.matmul(self.C, np.matmul(self.P, self.C.T))+ self.R
        K = np.matmul(np.matmul(self.P,self.C.T),np.linalg.inv(S))
        self.x_hat = self.x_hat + np.matmul(K, (z - np.matmul(self.C, self.x_hat)))
        self.P = np.matmul(np.identity(2) - np.matmul(K, self.C), self.P)


x_hat_hist = []
p_hist = []
z_hist = []
v_actual =[]
env = one_dim_env()
estimator = KALMANFilter()

u = np.concatenate([np.ones((30,)), np.zeros((40,)), -0.5*np.ones((30,))])
for i in range(100):
    z = env.external_step(u[i])

    estimator.prediction(u[i])

    estimator.measurement(z)

    x_hat_hist.append(estimator.x_hat)
    p_hist.append(estimator.P)
    z_hist.append(z)
    v_actual.append(env.x[1,:])

pos_hat = np.matmul(estimator.C, np.hstack(x_hat_hist)).squeeze(0)
vel_hat = np.hstack(x_hat_hist)[1,:]
sig_p = np.sqrt(np.vstack(p_hist).reshape([-1,2,2])[:,0,0])
sig_v = np.sqrt(np.vstack(p_hist).reshape([-1,2,2])[:,1,1])
z_np = np.hstack(z_hist).squeeze()
v_np = np.hstack(v_actual).squeeze()

con=3

plt.figure(figsize=(20, 20))

plt.subplot(211)
plt.plot(np.arange(0, 10, 0.1), pos_hat,"k-")
plt.plot(np.arange(0, 10, 0.1), pos_hat-con*sig_p,"r--")
plt.plot(np.arange(0, 10, 0.1), pos_hat+con*sig_p,"r--")
plt.plot(np.arange(0, 10, 0.1), z_np, "b-")
plt.ylabel("position")
plt.xlabel("time")

plt.subplot(212)
plt.plot(np.arange(0, 10, 0.1), vel_hat, "k-")
plt.plot(np.arange(0, 10, 0.1), vel_hat-con*sig_v, "r--")
plt.plot(np.arange(0, 10, 0.1), vel_hat+con*sig_v, "r--")
plt.plot(np.arange(0, 10, 0.1), v_np, "b-")
plt.plot(np.arange(0, 9.9, 0.1), (z_np[1:]-z_np[:-1])/0.1, "g-")
plt.ylabel("velocity")
plt.xlabel("time")



plt.show()



