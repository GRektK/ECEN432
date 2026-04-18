#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


def analog_to_thermometer25(x, Vfs, a, b, c, d):
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


def analog_to_thermometer2(x, Vfs, a, b, c, d):
    Vr=Vfs/2
    comparator_references = np.array([-Vr,0,Vr])/2+Vr
    #3 reference levels
    #3 comparators
    thermometer = np.array([])
    for i in range(3):
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


# In[4]:


def thermometer_to_binary25(thermometer):
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


# In[5]:


def thermometer_to_binary2(thermometer):
    # I'm going to implement boolean algebra to determine the code
    # using the ROM presented in the lecture slides.
    # Note: This implementation is sensitive to bubbles.
    b0=0
    b1=0
    if thermometer[2]:
        b0=1
        b1=1
    if not(thermometer[2]) and thermometer[1]:
        b1=1        
    if not(thermometer[1]) and thermometer[0]:
        b0=1
    
    return(f"{b1}{b0}")


# In[6]:


N=1024
Vin=np.linspace(0,1,N)
plt.figure()
for _ in range(100):

    a = np.random.uniform(-0.05,0.05,size=6)
    b = np.random.uniform(-0.05,0.05,size=6)
    c = np.random.uniform(-0.05,0.05,size=6)
    d = np.random.uniform(-0.05,0.05,size=6)
    x = Vin
    thermometer_codes=[analog_to_thermometer25(x_i,1,a,b,c,d) for x_i in x]
    binary_codes=[thermometer_to_binary25(thermometer_code_i) for
                  thermometer_code_i in thermometer_codes]

    plt.step(Vin, binary_codes,'k',linewidth=1,alpha=0.2,
             label="Monte Carlo trials" if _ == 0 else "")
    
# Plot ideal
ideal = np.zeros(6)
thermometer_codes=[analog_to_thermometer25(Vin_i,1,ideal,ideal,ideal,ideal) for Vin_i in Vin]
binary_codes=[thermometer_to_binary25(thermometer_code_i) for
              thermometer_code_i in thermometer_codes]

plt.step(Vin, binary_codes,'r', linewidth=2, label='Ideal transfer curve')

plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel(r"$D_{out}$")
plt.title("Transfer curve for 2.5 bit flash ADC with redundancy")
plt.grid()
plt.legend()
plt.show()


# In[7]:


#Compute DNL & INL here
a = np.zeros(6)
b = np.zeros(6)
c = np.zeros(6)
d = np.zeros(6)
N=1024
Vin=np.linspace(0,1,N)

plt.figure()

thermometer_codes=[analog_to_thermometer25(Vin_i,1,a,b,c,d) for Vin_i in Vin]
binary_codes_ideal=[thermometer_to_binary25(thermometer_code_i) for
                    thermometer_code_i in thermometer_codes]

a = np.random.uniform(-0.05,0.05,6)
thermometer_codes=[analog_to_thermometer25(Vin_i,1,a,b,c,d) for Vin_i in Vin]
binary_codes_error=[thermometer_to_binary25(thermometer_code_i) for
                    thermometer_code_i in thermometer_codes]

plt.step(Vin, binary_codes_ideal,'r',label='Ideal transfer curve')
plt.step(Vin, binary_codes_error,'k',label="Actual transfer curve")
plt.xlabel(r"$V_{in}\ [V_{fs}]$")
plt.ylabel(r"$D_{out}$")
plt.title("Transfer curve for 2.5 bit flash ADC with redundancy\n comparator offest")
plt.grid()
plt.legend()
plt.show()


# In[12]:


DNL = []
for target_code in range(7):
    idx_ideal_1 = np.where(np.array(binary_codes_ideal) == f"{target_code:03b}")[0][0]
    idx_ideal_2 = np.where(np.array(binary_codes_ideal) == f"{target_code:03b}")[0][-1]
    idx_error_1 = np.where(np.array(binary_codes_error) == f"{target_code:03b}")[0][0]
    idx_error_2 = np.where(np.array(binary_codes_error) == f"{target_code:03b}")[0][-1]
    
    DNL.append(Vin[idx_error_2]-Vin[idx_error_1]-Vin[idx_ideal_2]+Vin[idx_ideal_1])

INL=[]
for i in range(6):
    INL.append(sum(DNL[0:i+1]))
print(DNL)
print(INL)


# In[18]:


N=4096
T=1/1e7
f_sig=1/T/N*13
t = np.linspace(0,N*T,N, endpoint=False)
Vin = 0.5+0.5*np.sin(2*np.pi*f_sig*t)

SNDR25s=[]
SNDR2s=[]
for _ in range(50):
    a = np.random.uniform(-0.05,0.05,size=6)
    b = np.random.uniform(-0.05,0.05,size=6)
    c = np.random.uniform(-0.05,0.05,size=6)
    d = np.random.uniform(-0.05,0.05,size=6)
    x = Vin
    
    thermometer_codes=[analog_to_thermometer25(x_i,1,a,b,c,d) for x_i in x]
    binary_codes=[thermometer_to_binary25(thermometer_code_i) for
                  thermometer_code_i in thermometer_codes]

    Vout = [int(binary_code,2)/6 for binary_code in binary_codes]
    
    xf = scipy.fft.fft(Vout)

    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2+np.abs(xf[peak_idx+4])**2

    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDR25s.append(SNDR)

for _ in range(50):
    a = np.random.uniform(-0.05,0.05,size=3)
    b = np.random.uniform(-0.05,0.05,size=3)
    c = np.random.uniform(-0.05,0.05,size=3)
    d = np.random.uniform(-0.05,0.05,size=3)
    x = Vin
    
    thermometer_codes=[analog_to_thermometer2(x_i,1,a,b,c,d) for x_i in x]
    binary_codes=[thermometer_to_binary2(thermometer_code_i) for
                  thermometer_code_i in thermometer_codes]

    Vout = [int(binary_code,2)/4 for binary_code in binary_codes]
    
    xf = scipy.fft.fft(Vout)

    peak_idx = np.argmax(np.abs(xf[1:N//2])) + 1
    signal_power = np.abs(xf[peak_idx])**2+np.abs(xf[peak_idx+4])**2

    total_power = np.sum((np.abs(xf[1:N//2]))**2)
    noise_power = total_power-signal_power
    SNDR = 10*np.log10(signal_power/noise_power)
    SNDR2s.append(SNDR)


# In[20]:


plt.hist(SNDR25s,bins=np.arange(0,18,0.2),label="2.5 bits")
plt.hist(SNDR2s,bins=np.arange(0,18,0.2),label="2 bits")
plt.xlim([0,18])
plt.title("Monte Carlo simulation of 2.5 bit ADC")
plt.xlabel("SNDR [dB]")
plt.legend()
plt.show()

ENOB25s = [(SNDR25-1.76)/6.02 for SNDR25 in SNDR25s]
ENOB2s = [(SNDR2-1.76)/6.02 for SNDR2 in SNDR2s]
plt.hist(ENOB25s,bins=np.arange(0,3,0.03),label="2.5 bits")
plt.hist(ENOB2s,bins=np.arange(0,3,0.03),label="2 bits")
plt.xlim([0,3])
plt.title("Monte Carlo simulation of 2.5 bit ADC")
plt.xlabel("ENOB")
plt.legend()
plt.show()

