#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


def analog_to_thermometer(x, Vfs, a, b, c, d):
    Vr=Vfs/2
    comparator_references = np.array([-5*Vr,-3*Vr,-Vr,Vr,3*Vr,5*Vr])/8+Vr
    #6 reference levels
    #6 comparators
    thermometer = np.array([])
    for i in range(6):
        reference = comparator_references[i]
        offset=a[i]
        gain=b[i]
        second=c[i]
        third=d[i]
        
        Vin = x-reference
        Vin = offset + (1+gain)*(Vin) + second * Vin**2 + third * Vin**3
        if Vin>0:
            thermometer = np.append(thermometer,1)
        else:
            thermometer = np.append(thermometer,0)
    return thermometer


# In[3]:


def thermometer_to_binary(thermometer):
    # I'm going to implement boolean algebra to determine the code
    # using the ROM presented in the lecture slides.
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


# In[4]:


def ideal_stage(Vin, Vfs):
    ideal = np.zeros(6)
    thermometer = analog_to_thermometer(Vin,Vfs, ideal, ideal, ideal, ideal) #Comparators
    binary = thermometer_to_binary(thermometer) #Encoder
    Vres = Vin-int(binary,2)*Vfs/8 #DAC and subtraction 
    Vout = 4*Vres #Residue Amplifier
    
    return binary, Vout


# In[5]:


def ideal_ADC(Vin, Vfs):
    binary2, V2=ideal_stage(Vin,Vfs)
    binary1, V1=ideal_stage(V2, Vfs)
    binary0, V0=ideal_stage(V1,Vfs)
    code = int(binary2,2)*16+int(binary1,2)*4+int(binary0,2)
    return(code)


# In[6]:


Vin = np.linspace(0,1,1024)
decimal_codes = [ideal_ADC(Vin_i, 1) for Vin_i in Vin]
plt.step(Vin, decimal_codes)
plt.title("Ideal 3-stage pipeline ADC transfer curve\n 2.5 bits with redundancy")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel("Output Code")
plt.grid()
plt.show()


# In[7]:


N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

Vout = [ideal_ADC(Vin_i, 1)/126 for Vin_i in Vin]

plt.plot(t[:512]*1e6,Vin[:512],label=r"$V_{in}$")
plt.step(t[:512]*1e6,Vout[:512], label=r"$V_{out}$")
plt.xlabel(r"Time [$\mu s$]")
plt.ylabel(r"Voltage [$V_{fs}$]")
plt.grid()
plt.legend()
plt.title("Test tone input and output waveforms")
plt.show()


# In[10]:


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


# In[9]:


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

