from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import math

alp = np.array([2.0, 1.0, 0.5])
gam = np.array([5.0/3, 1.4, 1.2])
lam = np.array([[0.43558,0.22335,0.11410], [0.39334,0.20214,0.10352], [0.33090,0.17040,0.08749]])
ita = np.linspace(0, 1.0, 100)

def fgh_model(y, ita, alp, gam, lam):
    f, g, h = y
    df_di = (lam*f**2/(1+lam) - (ita*g*(alp-(gam+2)*lam)*f)/((1+lam)*h*(1-ita*f)) - (g*(alp-2*lam))/(h*(1+lam))) / ((1-ita*f)-(ita**2*g*lam/(h*(1-ita*f))))
    dg_di = (g/(1-ita*f))*(lam*ita*df_di - (alp-(gam+2)*lam)*f/(1+lam))
    dh_di = (h/(1-ita*f))*(ita*df_di - (alp-lam)*f/(1+lam))
    return [df_di, dg_di, dh_di]

#Initial conditions when gamma = 5/3
f0 = np.array([0.6439,0.6900,0.7177])
g0 = np.array([0.8542,0.8118,0.7804])
h0 = np.array([12.8868,8.0716,5.7506])

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
f0 = np.array([0.6781,0.7450,0.7851])
g0 = np.array([1.2134,1.0461,0.9457])
h0 = np.array([27.2242,15.5694,10.2758])

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
f0 = np.array([0.6884,0.7774,0.8368])
g0 = np.array([1.1691,1.4362,1.9585])
h0 = np.array([74.1596,36.0102,21.8676])

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
