# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:37:10 2020

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
from numpy import sin, cos, pi, arange
from numpy.random import randint

#%% NoisySignal.csv

fs = 1e6
Ts = 1/fs
t_end = 50e-3

t = arange(0,t_end-Ts,Ts)

f1 = 1.8e3
f2 = 1.9e3
f3 = 2e3
f4 = 1.85e3
f5 = 1.87e3
f6 = 1.94e3
f7 = 1.92e3

info_signal = 2.5*cos(2*pi*f1*t) + 1.75*cos(2*pi*f2*t) + 2*cos(2*pi*f3*t) + 2*cos(2*pi*f4*t) + 1*cos(2*pi*f5*t) + 1*cos(2*pi*f6*t) + 1.5*cos(2*pi*f7*t)

N = 25
my_sum = 0

for i in range(N+1):
    noise_amp     = 0.075*randint(-10,10,size=(1,1))
    noise_freq    = randint(-1e6,1e6,size=(1,1))
    noise_signal  = my_sum + noise_amp * cos(2*pi*noise_freq*t)
    my_sum = noise_signal

f6 = 50e3    #50kHz
f7 = 49.9e3
f8 = 51e3

pwr_supply_noise = 1.5*sin(2*pi*f6*t) + 1.25*sin(2*pi*f7*t) + 1*sin(2*pi*f8*t)

f9 = 60

low_freq_noise = 1.5*sin(2*pi*f9*t)

total_signal = info_signal + noise_signal + pwr_supply_noise + low_freq_noise
total_signal = total_signal.reshape(total_signal.size)

# plt.figure(figsize=(12,8))
# plt.subplot(3,1,1)
# plt.plot(t,info_signal)
# plt.grid(True)
# plt.subplot(3,1,2)
# plt.plot(t,info_signal+pwr_supply_noise)
# plt.grid(True)
# plt.subplot(3,1,3)
# plt.plot(t,total_signal)
# plt.grid(True)
# plt.show()

df = pd.DataFrame({'0':t,
                   '1':total_signal})

df.to_csv('NoisySignal.csv')

#%% Starter code

df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (12, 8))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [v]')
plt.show()

#%% Task 1- Identify magnitudes and frequencies due to low frequency vibrations

def clean_FFT(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = fft(x)  #perform the fast Fourier transform (fft)
    X_fft_shift = fftshift(X_fft)   # shift zero frequency components
                                    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
                                     # signal, (fs is the sampling frequency and
                                     # needs to be defined perviously in your code)
    X_mag = np.abs(X_fft_shift)/N # compute the magnitrudes of the signal
    X_phi = np.angle(X_fft_shift) # compute the phases of the signal
    
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:    # Allows the phase graph to be readable by setting larger values to 0
            X_phi[i] = 0
            
    return X_mag, X_phi, freq

def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])

x1 = sensor_sig
X_mag, X_phi, freq = clean_FFT(x1, fs)    # Runs x1 through the FFT

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize = (10,7))
plt.subplot(ax1)
make_stem(ax1, t, x1)
plt.grid()
plt.title('Task1')
plt.ylabel('x1(t)')
plt.xlabel('t[s]')
plt.xscale('log')

plt.subplot(ax2)
make_stem(ax2, freq, X_mag)
plt.grid()
plt.xscale('log')
plt.ylabel('|X1(f)|')
plt.subplot(ax3)
make_stem(ax3, freq, X_mag)
plt.xlim([1800, 2000])
plt.xscale('log')
plt.grid()

plt.subplot(ax4)
make_stem(ax4, freq, X_phi)
plt.grid()
plt.ylabel('/_ X1(f)')
plt.xlabel('f[Hz]')
plt.xscale('log')
plt.subplot(ax5)
make_stem(ax5, freq, X_phi)
plt.xlim([1800, 2000])
plt.grid()
plt.xlabel('f[Hz]')
plt.xscale('log')
plt.show()


#%% Task 2- Designing a working band-pass filter

R = 200
L = 1
C = 278e-9

num = [(1/(R*C)), 0]
den = [1, (1/(R*C)), (1/(L*C))]

steps = 1e1
w = np.arange(0, 1e6 + steps, steps) # Defining omega (need to check it)

omega, mag, phase = sig.bode((num, den), w)

fig = plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(omega, mag)
plt.ylabel('|H| dB')
plt.grid()
plt.title('Band-Pass Filter Magnitude')
plt.subplot(2,2,3)
plt.semilogx(omega, mag)
plt.ylabel('|H| dB')
plt.grid()
plt.xlim([0, 10e2])
plt.title('Zoom in for low-pass magnitude')
plt.subplot(2,2,4)
plt.semilogx(omega, mag)
plt.ylabel('|H| dB')
plt.grid()
plt.xlim(6e3, 10e5)
plt.title('Zoom in for high-pass magnitude')

fig = plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(omega, phase)
plt.yticks([-90, -45, 0, 45, 90])
plt.xlabel('w (rad/sec)')
plt.ylim([-90, 90])
plt.ylabel('/_ H (degrees)')
plt.title('Band-Pass Filter Phase')
plt.grid()
plt.subplot(2,2,3)
plt.semilogx(omega, phase)
plt.yticks([-90, -45, 0, 45, 90])
plt.xlabel('w (rad/sec)')
plt.ylim([-90, 90])
plt.title('Zoom in for low-pass phase')
plt.xlim([0, 10e1])
plt.grid()
plt.subplot(2,2,4)
plt.semilogx(omega, phase)
plt.yticks([-90, -45, 0, 45, 90])
plt.xlabel('w (rad/sec)')
plt.ylim([-90, 90])
plt.title('Zoom in for high-pass phase')
plt.xlim([6e3, 10e5])
plt.grid()
plt.show()


#%% Part 3- Using the filter on the sensor

# Noisy Signal filtered through the band-pass filter
numz, denz = sig.bilinear(num, den)

y = sig.lfilter(numz, denz, x1)

fig = plt.figure(figsize = (10,7))
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t[s]')
plt.title('Filtered Noisy Signal')
plt.show()

X_mag, X_phi, freq = clean_FFT(y, fs)    # Runs x1 through the FFT

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize = (10,7))
plt.subplot(ax1)
make_stem(ax1, t, y)
plt.grid()
plt.title('Task5')
plt.ylabel('x1(t)')
plt.xlabel('t[s]')
plt.xscale('log')

plt.subplot(ax2)
make_stem(ax2, freq, X_mag)
plt.grid()
plt.xscale('log')
plt.ylabel('|X1(f)|')
plt.subplot(ax3)
make_stem(ax3, freq, X_mag)
plt.xlim([1800, 2000])
plt.xscale('log')
plt.grid()

plt.subplot(ax4)
make_stem(ax4, freq, X_phi)
plt.grid()
plt.ylabel('/_ X1(f)')
plt.xlabel('f[Hz]')
plt.xscale('log')
plt.subplot(ax5)
make_stem(ax5, freq, X_phi)
plt.xlim([1800, 2000])
plt.grid()
plt.xlabel('f[Hz]')
plt.xscale('log')
plt.show()
