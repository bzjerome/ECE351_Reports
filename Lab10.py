# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:29:32 2020

@author: bzjer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import control as con
import pandas as pd
import time
from scipy.fftpack import fft, fftshift

#%% Function for adjusting the degrees

def fixDeg(Deg):
    for i in range(len(w)):
        if Deg[i] > 90:
            Deg[i] = Deg[i] - 180
    
    return Deg

#%% Part 1 Task 1

R = 1000
L = 27e-3
C = 100e-9

steps = 1e3
w = np.arange(1e3, 1e6 + steps, steps) # Defining omega (need to check it)

Mag = (20*np.log10((w/(R*C))/(np.sqrt(w**4 + (1/(R*C)**2 - 2/(L*C))*w**2 +
                       (1/(L*C))**2))))

Deg = (np.pi/2 - np.arctan((1/(R*C)*w)/(-w**2 + 1/(L*C))))*180/np.pi
Deg = fixDeg(Deg)       # Used to adjust the phase so that it's readable

# Bode Plots

fig = plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w, Mag)
plt.ylabel('|H(jw)| dB')
plt.title('My bode')
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(w, Deg)
plt.yticks([-90, -45, 0, 45, 90])
plt.xlabel('w (rad/sec)')
plt.ylim([-90, 90])
plt.ylabel('/_ H (degrees)')
plt.grid()
plt.show()

#%% Part 1 Task 2

num = [(1/(R*C)), 0]
den = [1, (1/(R*C)), (1/(L*C))]

omega, mag, phase = sig.bode((num, den), w)

fig = plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(omega, mag)
plt.ylabel('|H| dB')
plt.grid()
plt.title('Bode Plot Using sig.bode()')

plt.subplot(2,1,2)
plt.semilogx(omega, phase)
plt.yticks([-90, -45, 0, 45, 90])
plt.xlabel('w (rad/sec)')
plt.ylim([-90, 90])
plt.ylabel('/_ H (degrees)')
plt.grid()
plt.show()

#%% Part 1 Task 3

sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)

#%% Part 2 Task 1

fs = 1e5
T = 1/fs
t = np.arange(0, 1e-2, T)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

fig = plt.figure(figsize = (10,7))
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Plotted signal x(t)')
plt.show()

#%% Part 2 Task 4

numz, denz = sig.bilinear(num, den)

y = sig.lfilter(numz, denz, x)

fig = plt.figure(figsize = (10,7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t[s]')
plt.title('Filtered Signal y(t)')
plt.show()