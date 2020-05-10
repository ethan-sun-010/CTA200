#!/usr/bin/env python
import numpy
import math
import matplotlib.pyplot as plt

c_converge_real = []
c_converge_imag = []
c_diverge_real = []
c_diverge_imag = []
diverge_iteration_number = []

for x in numpy.arange(-2.0, 2.0, 0.05):
    for y in numpy.arange(-2.0, 2.0, 0.05):
        c = x + y*1j
        z = 0
        a = 1
        for i in range(25):
            z = z**2 + c
            if numpy.isfinite(z.real) and numpy.isfinite(z.imag):
                a = a + 1
        if numpy.isfinite(z.real) and numpy.isfinite(z.imag):
            c_converge_real.append(c.real)
            c_converge_imag.append(c.imag)
        else:
            c_diverge_real.append(c.real)
            c_diverge_imag.append(c.imag)
            diverge_iteration_number.append(a)

conv = plt.scatter(c_converge_real,c_converge_imag, s = 10, c = "red", label = "convergent points")
div = plt.scatter(c_diverge_real,c_diverge_imag, s = 10, c=diverge_iteration_number, label = "divergent points")
plt.colorbar()
plt.legend()
plt.show()
