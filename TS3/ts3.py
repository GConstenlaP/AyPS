# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:44:54 2026

@author: Usuario
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy



# %% definiciones

N = 1000

fs = 1000
 
n = np.arange(N)

t = n / fs

delta_f = fs / N  # resolución espectral en hz = 1

#1 f0 = k0∗fS/N   =  k0.Δf

# %% funciones

def mi_funcion_sen(vmax=np.sqrt(2), dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(nn) / fs

    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

    return tt, xx



def graficar_primer_ciclo(t, x, f, titulo):
    
    #implemento redondeo hacia arriba copn np.ceil
    muestras_por_ciclo = int(np.ceil(fs / f)) #muestras que componen un ciclo

    plt.figure()
    plt.plot(t[:muestras_por_ciclo + 1], x[:muestras_por_ciclo + 1],'o-')
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid()
    plt.show()

# %% mi script
#Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál es el efecto de dicha desintonía en el espectro visualizado.

# %% 1 k0= n/4

k0 = N/4
k1 = N/4 + 0.25
k2 = N/4 + 0.5

f0 = k0 * (fs/N)
f1 = k1 * (fs/N)
f2 = k2 * (fs/N)

t0, x0 = mi_funcion_sen(dc=0, ff=f0, ph=0, nn=N, fs=fs)

t1, x1 = mi_funcion_sen(dc=0, ff=f1, ph=0, nn=N, fs=fs)
t2, x2 = mi_funcion_sen(dc=0, ff=f2, ph=0, nn=N, fs=fs)


print("la frecuencia de la señal f0 es :", f0)

print("la frecuencia de la señal f2 es :", f1)

print("la frecuencia de la señal f2 es :", f2)



graficar_primer_ciclo(t0, x0, f0, 'Primer ciclo de Señal 1')

graficar_primer_ciclo(t1, x1, f1, 'Primer ciclo de Señal 2')

graficar_primer_ciclo(t2, x2, f2, 'Primer ciclo de Señal 3')



# %% f02

# Calculo de Potencia:
# modulo_fft_X   = np.abs(scipy.fft.fft(X)[:N//2]) / N
# Max1 = np.max(modulo_fft_X)
# PSD_X   = (modulo_fft_X /Max1)**2
# PSD_X_db   = 10 * np.log10(PSD_X)

fft_0 =np.abs( scipy.fft.fft(x0)[:N//2])/N
Max1 = np.max(fft_0 )
PSD_X0   = (fft_0/Max1)**2
PSD_X0_db   = 10 * np.log10(PSD_X0)

fft_1 =np.abs( scipy.fft.fft(x1)[:N//2])/N
PSD_X1   = (fft_1/Max1)**2
PSD_X1_db   = 10 * np.log10(PSD_X1)

fft_2 =np.abs( scipy.fft.fft(x2)[:N//2])/N
PSD_X2   = (fft_2/Max1)**2
PSD_X2_db   = 10 * np.log10(PSD_X2)

frecuencias = np.arange(0, fs/2, fs/N)

plt.figure()
plt.plot(frecuencias, PSD_X0_db, color= 'red', label  = f'k0 = {k0}hz')
plt.plot(frecuencias, PSD_X1_db, color= 'blue', label  = f'k1 = {k1}hz')
plt.plot(frecuencias, PSD_X2_db, color= 'yellow', label  = f'k2 = {k2}hz')

plt.title('Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-100, 5)
plt.legend(loc='best')
plt.show()


plt.figure()
plt.plot(frecuencias, PSD_X0_db, color= 'red', label  = f'k0 = {k0}hz')
plt.plot(frecuencias, PSD_X1_db, color= 'blue', label  = f'k1 = {k1}hz')
plt.plot(frecuencias, PSD_X2_db, color= 'yellow', label  = f'k2 = {k2}hz')

plt.title('Tercer Gráfica: Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-330, 5)
plt.legend(loc='best')
plt.show()


plt.figure()
plt.plot(frecuencias, PSD_X0_db, color= 'red', label  = f'k0 = {k0}hz')
plt.plot(frecuencias, PSD_X1_db, color= 'blue', label  = f'k1 = {k1}hz')
plt.plot(frecuencias, PSD_X2_db, color= 'yellow', label  = f'k2 = {k2}hz')

plt.title('Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-10, 5)
plt.legend(loc='best')
plt.show()