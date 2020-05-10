#!/usr/bin/env python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import math

"initial conditions"
S0 = 999.0
I0 = 1.0
R0 = 0
D0 = 0
N = 1000.0

"different beta, gamma, and mu values simulate different scenarios from moderate to dire"
beta = np.array([0.2, 0.4, 0.6, 0.8])
gamma = np.array([0.15, 0.1, 0.06, 0.03])
mu = np.array([0.01, 0.05, 0.1, 0.2])
t = np.linspace(0, 200, 200)

def SIR_model(y, t, beta, gamma, mu):
    S, I, R, D = y
    dS_dt = - (beta*S*I)/N
    dI_dt = (beta*S*I)/N - gamma*I - mu*I
    dR_dt = gamma*I
    dD_dt = mu*I
    return [dS_dt, dI_dt, dR_dt, dD_dt]

case_1 = (beta[0], gamma[0], mu[0])
result_raw = np.array(integrate.odeint(SIR_model, [S0, I0, R0, D0], t, args=(case_1)))
S = result_raw[:,0]
I = result_raw[:,1]
R = result_raw[:,2]
D = result_raw[:,3]
plt.figure(1)
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.plot(t, D, label="Death")
plt.legend()
plt.xlabel("time")
plt.ylabel("case count")

case_2 = (beta[1], gamma[1], mu[1])
result_raw = np.array(integrate.odeint(SIR_model, [S0, I0, R0, D0], t, args=(case_2)))
S = result_raw[:,0]
I = result_raw[:,1]
R = result_raw[:,2]
D = result_raw[:,3]
plt.figure(2)
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.plot(t, D, label="Death")
plt.legend()
plt.xlabel("time")
plt.ylabel("case count")

case_3 = (beta[2], gamma[2], mu[2])
result_raw = np.array(integrate.odeint(SIR_model, [S0, I0, R0, D0], t, args=(case_3)))
S = result_raw[:,0]
I = result_raw[:,1]
R = result_raw[:,2]
D = result_raw[:,3]
plt.figure(3)
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.plot(t, D, label="Death")
plt.legend()
plt.xlabel("time")
plt.ylabel("case count")

case_4 = (beta[3], gamma[3], mu[3])
result_raw = np.array(integrate.odeint(SIR_model, [S0, I0, R0, D0], t, args=(case_4)))
S = result_raw[:,0]
I = result_raw[:,1]
R = result_raw[:,2]
D = result_raw[:,3]
plt.figure(4)
plt.plot(t, S, label="Susceptible")
plt.plot(t, I, label="Infected")
plt.plot(t, R, label="Recovered")
plt.plot(t, D, label="Death")
plt.legend()
plt.xlabel("time")
plt.ylabel("case count")

plt.show()

