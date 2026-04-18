#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[161]:


def analog_to_thermometer(x, Vfs):
    Vr=Vfs/2
    comparator_references = np.array([-5*Vr,-3*Vr,-Vr,Vr,3*Vr,5*Vr])/8+Vr
    #6 reference levels for comparators, 7 DAC levels
    #6 comparators
    thermometer = np.array([])
    for reference in comparator_references:
        if x>reference:
            thermometer = np.append(thermometer,1)
        else:
            thermometer = np.append(thermometer,0)
    return thermometer


# In[162]:


def thermometer_to_binary(thermometer):
    # I'm going to implement boolean algebra to determine the code
    # Using the ROM presented in the lecture slides.
    # Note: This implementation is sensitive to bubbles.
    b0=0
    b1=0
    b2=0
    if thermometer[5]:
        b1=1
        b2=1
    if not(thermometer[5]) and thermometer[4]:
        b0=1
        b2=1
    if not(thermometer[4]) and thermometer[3]:
        b2=1
    if not(thermometer[3]) and thermometer[2]:
        b0=1
        b1=1
    if not(thermometer[2]) and thermometer[1]:
        b1=1        
    if not(thermometer[1]) and thermometer[0]:
        b0=1
    
    return(f"{b2}{b1}{b0}")


# In[218]:


N=1024

Vin=np.linspace(0,1,N)
thermometer_codes=[analog_to_thermometer(Vin_i,1) for Vin_i in Vin]
binary_codes=[thermometer_to_binary(thermometer_code_i) for
              thermometer_code_i in thermometer_codes]

plt.step(Vin, binary_codes, label="Ideal transfer curve")
plt.plot(np.arange(0,7)/6,np.arange(0,7),label="Reference curve")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel(r"$D_{out}$")
plt.title("Ideal transfer curve for 2.5 bit flash ADC with redundancy")
plt.grid()
plt.legend()
plt.show()


# In[216]:


quantization_error = [Vin[i]-int(binary_codes[i],2)/6 for i in range(N)]

plt.plot(Vin, quantization_error)
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel(r"$\varepsilon_q\ [V_{fs}]$")
plt.title("Quantization error for 2.5 bit flash ADC with redundancy")
plt.show()

print(f"Maximum quantization error: {np.max(quantization_error)}")
print(f"Minimum quantization error: {np.min(quantization_error)}")


# In[241]:


N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

thermometer_codes=[analog_to_thermometer(Vin_i,1) for Vin_i in Vin]
binary_codes=[thermometer_to_binary(thermometer_code_i) for
              thermometer_code_i in thermometer_codes]
Vout = [int(binary_code,2)/6 for binary_code in binary_codes]

plt.plot(t[:512]*1e6,Vin[:512],label=r"$V_{in}$")
plt.step(t[:512]*1e6,Vout[:512], label=r"$V_{out}$")
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"Voltage [$V_{fs}$]")
plt.grid()
plt.legend()
plt.title("Test tone input and output waveforms")
plt.show()


# In[244]:


xf = scipy.fft.fft(Vout)
xf_max = max(np.abs(xf))
xf_db = 20*np.log10(np.abs(xf)/xf_max+1e-12)
f = scipy.fft.fftfreq(N, T)[:N//2]
plt.semilogx(f, xf_db[0:N//2])
plt.title(r"PSD of $V_{out}$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Power [dB]")
plt.ylim([-100,5])
plt.show()


# In[226]:


peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
signal_power = np.abs(xf[peak_idx])**2

total_power = np.sum((np.abs(xf[1:N//2]))**2)
noise_power = total_power-signal_power

#print(total_power)
#print(signal_power)
#print(noise_power)

SNDR = 10*np.log10(signal_power/noise_power)
print(f"SNDR: {SNDR}")
print(f"ENOB: {(SNDR-1.67)/6.02}")

