#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


def analog_to_thermometer(x, Vfs, comparators):
    [a,b,c,d]=comparators
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


def stage(Vin, Vfs,comparators, RA):
    thermometer = analog_to_thermometer(Vin,Vfs, comparators) #Comparators
    binary = thermometer_to_binary(thermometer) #Encoder
    Vres = Vin-int(binary,2)*Vfs/8 #DAC and subtraction 
    Vout = (1+RA[0])*4*Vres+RA[1]+RA[2]*Vres**2+RA[3]*Vres**3 #Residue Amplifier
    return binary, Vout


# In[5]:


def ADC(Vin, Vfs,comparators,RA):
    binary2, V2=stage(Vin,Vfs,comparators[0], RA[0])
    binary1, V1=stage(V2, Vfs,comparators[1], RA[1])
    binary0, V0=stage(V1, Vfs,comparators[2], RA[2])
    code = int(binary2,2)*64+int(binary1,2)*8+int(binary0,2)
    return(code)


# In[16]:


comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

Vin = np.linspace(0,1,1024)
decimal_codes = [ADC(Vin_i, 1, comparators, RA) for Vin_i in Vin]
plt.step(Vin, decimal_codes)
plt.title("Ideal 3-stage pipeline ADC transfer curve\n 2.5 bits without redundancy")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel("Output Code")
plt.grid()
plt.show()


# In[6]:


#1st-stage sub-ADC offset voltage

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(-0.4,0.4,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for offset_voltage in sweep_values:
    comparators[0][0]=np.ones(6)*offset_voltage
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 1st-stage sub-ADC $V_{offset}$")
plt.xlabel(r"$V_{offset} [V_{fs}]$")
plt.ylabel("SNDR [dB]")
plt.show()


# In[7]:


#2nd-stage sub-ADC gain error

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(0,0.05,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for gain_error in sweep_values:
    comparators[1][1]=np.ones(6)*gain_error
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 2nd-stage sub-ADC $\epsilon$")
plt.xlabel(r"$\epsilon [V/V]$")
plt.ylabel("SNDR [dB]")
plt.show()


# In[8]:


#3rd-stage sub-ADC second-order nonlinearity

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(-0.05,0.05,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for alpha2 in sweep_values:
    comparators[2][2]=np.ones(6)*alpha2
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 3rd-stage sub-ADC $\alpha_2$")
plt.xlabel(r"$\alpha_2 [\frac{1}{V_{fs}}]$")
plt.ylabel("SNDR [dB]")
plt.show()


# In[9]:


#1st-stage RA gain error

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(0,0.05,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for gain_error in sweep_values:
    RA[0][0]=gain_error
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 1st-stage RA $\epsilon$")
plt.xlabel(r"$\epsilon [V/V]$")
plt.ylabel("SNDR [dB]")
plt.show()


# In[10]:


#2nd-stage RA offset voltage

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(-0.4,0.4,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for offset_voltage in sweep_values:
    RA[1][1]=offset_voltage
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 2nd-stage RA $V_{offset}$")
plt.xlabel(r"$V_{offset} [V_{fs}]$")
plt.ylabel("SNDR [dB]")
plt.show()


# In[11]:


#3rd-stage RA third-order nonlinearity

comparators=[]
for _ in range(3):
    comparators.append([np.zeros(6), #Offset
                        np.zeros(6), #Gain
                        np.zeros(6), #Second
                        np.zeros(6)])#Third

RA=[]
for _ in range(3):
    RA.append(np.zeros(4)) #Gain, Offset, Second, Third

sweep_values = np.linspace(-0.05,0.05,512)

SNDRs = []
N=1024
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)
for alpha3 in sweep_values:
    RA[2][3]=alpha3
    Vout = [ADC(Vin_i, 1, comparators, RA)/438 for Vin_i in Vin]

    xf = scipy.fft.fft(Vout)
    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2
    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDRs.append(SNDR)
    
plt.plot(sweep_values, SNDRs)
plt.title("3-stage pipeline ADC transfer curve\n"r"Parameter sweep on 3rd-stage RA $\alpha_3$")
plt.xlabel(r"$\alpha_3 [/frac{1}{V_{fs}^2}]$")
plt.ylabel("SNDR [dB]")
plt.show()

