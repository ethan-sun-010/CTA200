from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import math

alp = np.array([2.0, 1.0, 0.5])
gam = np.array([5.0/3, 1.4, 1.2])
lam = np.array([[0.47216,0.23608,0.11804], [0.43050,0.21525,0.10763], [0.36602,0.18301,0.09151]])
ita = np.linspace(0, 1.0, 100)

def fgh_model(y, ita, alp, gam, lam):
    f, g, h = y
    df_di = (lam*f**2/(1+lam) - (ita*g*(alp-(gam+2)*lam)*f)/((1+lam)*h*(1-ita*f)) - (g*(alp-2*lam))/(h*(1+lam))) / ((1-ita*f)-(ita**2*g*lam/(h*(1-ita*f))))
    dg_di = (g/(1-ita*f))*(lam*ita*df_di - (alp-(gam+2)*lam)*f/(1+lam))
    dh_di = (h/(1-ita*f))*(ita*df_di - (alp-lam)*f/(1+lam))
    return [df_di, dg_di, dh_di]

#Initial conditions when gamma = 5/3
f0 = np.array([0.6446,0.6940,0.7237])
g0 = np.array([0.8568,0.8141,0.7860])
h0 = np.array([13.7680,8.6313,6.1976])

#Plotting gamma = 5/3, alpha = 2
case_1 = (alp[0], gam[0], lam[0][0])
result = np.array(integrate.odeint(fgh_model, [f0[0], g0[0], h0[0]], ita, args=(case_1)))                
f1 = result[:,0]
g1 = result[:,1]
h1 = result[:,2]
#plot1 = plt.figure(1)
fig, ax1 = plt.subplots()
ax1.set_xlabel("ita")
ax1.set_ylabel("f, g")
plt.ylim(0.0, 1.0)
ax1.plot(ita, f1, '-', color='r', label="f, alpha=2")
ax1.plot(ita, g1, '-', color='b', label="g, alpha=2")
ax2 = ax1.twinx()
ax2.set_ylabel("h")
plt.ylim(0.0, 15.0)
ax2.plot(ita, h1, '-', color='g', label="h, alpha=2")
fig.tight_layout(pad=4)

#Plotting gamma = 5/3, alpha = 1
case_2 = (alp[1], gam[0], lam[0][1])
result = np.array(integrate.odeint(fgh_model, [f0[1], g0[1], h0[1]], ita, args=(case_2)))               
f2 = result[:,0]
g2 = result[:,1]
h2 = result[:,2]
ax1.plot(ita, f2, ':', color='r', label="f, alpha=1")
ax1.plot(ita, g2, ':', color='b', label="g, alpha=1")
ax2.plot(ita, h2, ':', color='g', label="h, alpha=1")

#Plotting gamma = 5/3, alpha = 1/2
case_3 = (alp[2], gam[0], lam[0][2])
result = np.array(integrate.odeint(fgh_model, [f0[2], g0[2], h0[2]], ita, args=(case_3)))               
f3 = result[:,0]
g3 = result[:,1]
h3 = result[:,2]
ax1.plot(ita, f3, '--', color='r', label="f, alpha=1/2")
ax1.plot(ita, g3, '--', color='b', label="g, alpha=1/2")
ax2.plot(ita, h3, '--', color='g', label="h, alpha=1/2")

plt.margins(x=0)
fig.legend(loc='upper left',ncol=3, mode="expand", borderaxespad=0.5, prop={'size': 7})
plt.show()

############################################################################################

#Initial conditions when gamma = 7/5
f0 = np.array([0.6808,0.7444,0.7867])
g0 = np.array([1.2118,1.0439,0.9448])
h0 = np.array([27.3695,15.4804,10.3165])

#Plotting gamma = 7/5, alpha = 2
case_4 = (alp[0], gam[1], lam[1][0])
result = np.array(integrate.odeint(fgh_model, [f0[0], g0[0], h0[0]], ita, args=(case_4)))                
f1 = result[:,0]
g1 = result[:,1]
h1 = result[:,2]
#plot2 = plt.figure(2)
fig, ax1 = plt.subplots()
ax1.set_xlabel("ita")
ax1.set_ylabel("f, g")
plt.ylim(0.0, 1.6)
ax1.plot(ita, f1, '-', color='r', label="f, alpha=2")
ax1.plot(ita, g1, '-', color='b', label="g, alpha=2")
ax2 = ax1.twinx()
ax2.set_ylabel("h")
plt.ylim(0.0, 28.0)
ax2.plot(ita, h1, '-', color='g', label="h, alpha=2")
fig.tight_layout(pad=4)

#Plotting gamma = 7/5, alpha = 1
case_5 = (alp[1], gam[1], lam[1][1])
result = np.array(integrate.odeint(fgh_model, [f0[1], g0[1], h0[1]], ita, args=(case_5)))               
f2 = result[:,0]
g2 = result[:,1]
h2 = result[:,2]
ax1.plot(ita, f2, ':', color='r', label="f, alpha=1")
ax1.plot(ita, g2, ':', color='b', label="g, alpha=1")
ax2.plot(ita, h2, ':', color='g', label="h, alpha=1")

#Plotting gamma = 7/5, alpha = 1/2
case_6 = (alp[2], gam[1], lam[1][2])
result = np.array(integrate.odeint(fgh_model, [f0[2], g0[2], h0[2]], ita, args=(case_6)))               
f3 = result[:,0]
g3 = result[:,1]
h3 = result[:,2]
ax1.plot(ita, f3, '--', color='r', label="f, alpha=1/2")
ax1.plot(ita, g3, '--', color='b', label="g, alpha=1/2")
ax2.plot(ita, h3, '--', color='g', label="h, alpha=1/2")

plt.margins(x=0)
fig.legend(loc='upper left',ncol=3, mode="expand", borderaxespad=0.5, prop={'size': 7})
plt.show()

############################################################################################

#Initial conditions when gamma = 6/5
f0 = np.array([0.7105,0.7865,0.8406])
g0 = np.array([1.9584,1.4401,1.1735])
h0 = np.array([74.2969,36.2441,21.7877])

#Plotting gamma = 6/5, alpha = 2
case_7 = (alp[0], gam[2], lam[2][0])
result = np.array(integrate.odeint(fgh_model, [f0[0], g0[0], h0[0]], ita, args=(case_7)))                
f1 = result[:,0]
g1 = result[:,1]
h1 = result[:,2]
fig, ax1 = plt.subplots()
ax1.set_xlabel("ita")
ax1.set_ylabel("f, g")
plt.ylim(0.0, 4.0)
ax1.plot(ita, f1, '-', color='r', label="f, alpha=2")
ax1.plot(ita, g1, '-', color='b', label="g, alpha=2")
ax2 = ax1.twinx()
ax2.set_ylabel("h")
plt.ylim(0.0, 80.0)
ax2.plot(ita, h1, '-', color='g', label="h, alpha=2")
fig.tight_layout(pad=4)

#Plotting gamma = 6/5, alpha = 1
case_8 = (alp[1], gam[2], lam[2][1])
result = np.array(integrate.odeint(fgh_model, [f0[1], g0[1], h0[1]], ita, args=(case_8)))               
f2 = result[:,0]
g2 = result[:,1]
h2 = result[:,2]
ax1.plot(ita, f2, ':', color='r', label="f, alpha=1")
ax1.plot(ita, g2, ':', color='b', label="g, alpha=1")
ax2.plot(ita, h2, ':', color='g', label="h, alpha=1")

#Plotting gamma = 6/5, alpha = 1/2
case_9 = (alp[2], gam[2], lam[2][2])
result = np.array(integrate.odeint(fgh_model, [f0[2], g0[2], h0[2]], ita, args=(case_9)))               
f3 = result[:,0]
g3 = result[:,1]
h3 = result[:,2]
ax1.plot(ita, f3, '--', color='r', label="f, alpha=1/2")
ax1.plot(ita, g3, '--', color='b', label="g, alpha=1/2")
ax2.plot(ita, h3, '--', color='g', label="h, alpha=1/2")

plt.margins(x=0)
fig.legend(loc='upper left',ncol=3, mode="expand", borderaxespad=0.5, prop={'size': 7})
plt.show()
