#!/usr/bin/env python
import numpy
import math
import matplotlib.pyplot as plt
import cmath


c_converge_real = []
c_converge_imag = []
c_diverge_real = []
c_diverge_imag = []

for x in numpy.arange(-2.0, 2.0, 0.01):
    for y in numpy.arange(-2.0, 2.0, 0.01):
        c = x + y*1j
        z = 0
        for i in range(100):
            z = z**2 + c
        if numpy.isfinite(z.real) and numpy.isfinite(z.imag):
            c_converge_real.append(c.real)
            c_converge_imag.append(c.imag)
        else:
            c_diverge_real.append(c.real)
            c_diverge_imag.append(c.imag)

plt.scatter(c_converge_real,c_converge_imag, color='red')
plt.scatter(c_diverge_real,c_diverge_imag, color='blue')
plt.show()


