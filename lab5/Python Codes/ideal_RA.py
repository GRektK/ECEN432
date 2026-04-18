#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[3]:


Vres = np.linspace(0,1,1024)
Vout = 4*Vres
plt.plot(Vres,Vout)
plt.xlabel(r"$V_{res}\ [V_{fs}]$")
plt.ylabel(r"$V_{out}\ [V_{fs}]$")
plt.title("4x residue amplifier transfer curve")
plt.grid()
plt.show()


# In[4]:


N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vres = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

Vout = 4*Vres

plt.plot(t[:512]*1e6,Vres[:512],label=r"$V_{res}$")
plt.step(t[:512]*1e6,Vout[:512], label=r"$V_{out}$")
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"Voltage [$V_{fs}$]")
plt.grid()
plt.legend()
plt.title("Test tone input and output waveforms")
plt.show()


# In[6]:


xf = scipy.fft.fft(Vout)
xf_max = max(np.abs(xf))
xf_db = 20*np.log10(np.abs(xf)/xf_max+1e-12)
f = scipy.fft.fftfreq(N, T)[:N//2]
plt.semilogx(f, xf_db[0:N//2])
plt.title(r"PSD of $V_{out}$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Power [dB]")
plt.ylim([-300,5])
plt.show()


# In[10]:


peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
signal_power = np.abs(xf[peak_idx])**2

total_power = np.sum((np.abs(xf[1:N//2]))**2)
noise_power = total_power-signal_power

#print(total_power)
#print(signal_power)
#print(noise_power)

SNDR = 10*np.log10(signal_power/noise_power)
print(f"SNDR: {SNDR}")
print(f"ENOB: {(SNDR-1.76)/6.02}")

