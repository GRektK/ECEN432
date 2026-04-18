#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


gain = 0.05
offset = 0.2
second = 0
third = 0

Vres = np.linspace(0,1,1024)
Vout = (1+gain)*4*Vres+offset+second*Vres**2+third*Vres**3
plt.plot(Vres,Vout,label="Actual curve")
plt.plot(Vres,4*Vres,label="Reference curve")
plt.xlabel(r"$V_{res}\ [V_{fs}]$")
plt.ylabel(r"$V_{out}\ [V_{fs}]$")
plt.title("4x residue amplifier transfer curve\n Setup 1")
plt.grid()
plt.legend()
plt.show()


# In[3]:


gain = 0.05
offset = 0.3
second = 0.02
third = 0.01

N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vres = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

Vout = (1+gain)*4*Vres+offset+second*Vres**2+third*Vres**3

plt.plot(t[:512]*1e6,Vres[:512],label=r"$V_{res}$")
plt.step(t[:512]*1e6,Vout[:512], label=r"$V_{out}$")
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"Voltage [$V_{fs}$]")
plt.grid()
plt.legend()
plt.title("Test tone input and output waveforms\n Setup 2")
plt.show()


# In[4]:


xf = scipy.fft.fft(Vout)
xf_max = max(np.abs(xf))
xf_db = 20*np.log10(np.abs(xf)/xf_max+1e-12)
f = scipy.fft.fftfreq(N, T)[:N//2]
plt.semilogx(f, xf_db[0:N//2])
plt.title(r"PSD of $V_{out}$"+"\n Setup 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Power [dB]")
plt.ylim([-100,5])

ax=plt.gca()
ax.annotate('HD2', xy=(f_sig*2-16e3, -55))
ax.annotate('HD3', xy=(f_sig*3-20e3, -80))

plt.show()


# In[5]:


sorted_amp = sorted(set(np.abs(xf[1:N//2])), reverse=True)
SFDR = 20*np.log10(sorted_amp[0])-20*np.log10(sorted_amp[1])
print(f"SFDR: {SFDR}")

peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
signal_power = np.abs(xf[peak_idx])**2+np.abs(xf[peak_idx+4])**2

total_power = np.sum((np.abs(xf[1:N//2]))**2)
noise_power = total_power-signal_power

#print(total_power)
#print(signal_power)
#print(noise_power)

SNDR = 10*np.log10(signal_power/noise_power)
print(f"SNDR: {SNDR}")
print(f"ENOB: {(SNDR-1.76)/6.02}")


# In[6]:


gain = 0.05
offset = 0.4
second = 0.05
third = 0.03

N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vres = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

Vout = (1+gain)*4*Vres+offset+second*Vres**2+third*Vres**3

plt.plot(t[:512]*1e6,Vres[:512],label=r"$V_{res}$")
plt.step(t[:512]*1e6,Vout[:512], label=r"$V_{out}$")
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"Voltage [$V_{fs}$]")
plt.grid()
plt.legend()
plt.title("Test tone input and output waveforms\n Setup 3")
plt.show()


# In[7]:


xf = scipy.fft.fft(Vout)
xf_max = max(np.abs(xf))
xf_db = 20*np.log10(np.abs(xf)/xf_max+1e-12)
f = scipy.fft.fftfreq(N, T)[:N//2]
plt.semilogx(f, xf_db[0:N//2])
plt.title(r"PSD of $V_{out}$"+"\n Setup 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Power [dB]")
plt.ylim([-100,5])

ax=plt.gca()
ax.annotate('HD2', xy=(f_sig*2-16e3, -50))
ax.annotate('HD3', xy=(f_sig*3-20e3, -73))

plt.show()


# In[8]:


sorted_amp = sorted(set(np.abs(xf[1:N//2])), reverse=True)
SFDR = 20*np.log10(sorted_amp[0])-20*np.log10(sorted_amp[1])
print(f"SFDR: {SFDR}")

peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
signal_power = np.abs(xf[peak_idx])**2+np.abs(xf[peak_idx+4])**2

total_power = np.sum((np.abs(xf[1:N//2]))**2)
noise_power = total_power-signal_power

#print(total_power)
#print(signal_power)
#print(noise_power)

SNDR = 10*np.log10(signal_power/noise_power)
print(f"SNDR: {SNDR}")
print(f"ENOB: {(SNDR-1.76)/6.02}")


# In[9]:


# Parameter sweep on alpha2
gain=0
offset=0
second=0
third=0
sweep_values = np.linspace(-1,1,1024)
HD2_values = []
N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vres = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for alpha2 in sweep_values:

    Vout = (1+gain)*4*Vres+offset+alpha2*Vres**2+third*Vres**3

    xf = scipy.fft.fft(Vout)

    fund = np.abs(xf[13])
    harm2 = np.abs(xf[26])
    HD2_values.append(20*np.log10(harm2 / fund))

plt.plot(sweep_values, HD2_values)
plt.xlabel(r"$\alpha_2\ [\frac{1}{V_{fs}}]$")
plt.ylabel("HD2 [dB]")
plt.title(r"Parameter sweep on $\alpha_2$")
plt.show()


# In[10]:


# Parameter sweep on alpha3
gain=0
offset=0
second=0
third=0
sweep_values = np.linspace(-1,1,1024)

HD3_values = []
N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vres = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for alpha3 in sweep_values:

    Vout = (1+gain)*4*Vres+offset+second*Vres**2+alpha3*Vres**3

    xf = scipy.fft.fft(Vout)
    fund = np.abs(xf[13])
    harm3 = np.abs(xf[39])
    HD3_values.append(20*np.log10(harm3 / fund))

plt.plot(sweep_values, HD3_values)
plt.xlabel(r"$\alpha_3\ [\frac{1}{V_{fs}^2}]$")
plt.ylabel("HD3 [dB]")
plt.title(r"Parameter sweep on $\alpha_3$")
plt.show()

