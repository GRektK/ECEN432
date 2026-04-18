#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy


# In[2]:


def MDAC(x, Vfs):
    Vr=Vfs/2
    reference = np.array([-5*Vr,-3*Vr,-Vr,Vr,3*Vr,5*Vr])/8+Vr
    match x:
        case n if n < reference[0]:
            Vout = 4*x
        case n if reference[0] <= n < reference[1]:
            Vout = 4*x - 1*Vr
        case n if reference[1] <= n < reference[2]:
            Vout = 4*x - 2*Vr
        case n if reference[2] <= n < reference[3]:
            Vout = 4*x - 3*Vr
        case n if reference[3] <= n < reference[4]:
            Vout = 4*x - 4*Vr
        case n if reference[4] <= n < reference[5]:
            Vout = 4*x - 5*Vr
        case n if n >= reference[5]:
            Vout = 4*x - 6*Vr
    
    return Vout


# In[3]:


Vin = np.linspace(0,1,1024)
Vout = [MDAC(Vin_i,1) for Vin_i in Vin]

plt.plot(Vin,Vout)
plt.xlabel("Input code")
plt.ylabel(r"Output voltage [$V_{fs}$]")
plt.title("Ideal 2.5-bit MDAC transfer curve")
plt.grid()
plt.show()

