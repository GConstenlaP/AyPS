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


plt.figure(figsize = (10,4))
plt.plot(frecuencias, PSD_X0_db, color= 'red', label  = f'f0 = {k0}hz', marker='o', markersize=1.5)
plt.plot(frecuencias, PSD_X1_db, color= 'blue', label  = f'f1 = {k1}hz', marker='o', markersize=1.5)
plt.plot(frecuencias, PSD_X2_db, color= 'yellow', label  = f'f2 = {k2}hz', marker='o', markersize=1.5)

plt.title('Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.show()

plt.figure(figsize = (10,4))
plt.plot(frecuencias, PSD_X0_db, color= 'red', label  = f'f0 = {k0}hz', ls = ':', linewidth=1, marker='o', markersize=1.5)
plt.plot(frecuencias, PSD_X1_db, color= 'blue', label  = f'f1 = {k1}hz', ls = ':', linewidth=1,        marker='o', markersize=1.5)
plt.plot(frecuencias, PSD_X2_db, color= 'yellow', label  = f'f2 = {k2}hz', ls = ':', linewidth=1,      marker='o', markersize=1.5)


plt.title('Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-100, 5)
plt.legend(loc='best')
plt.show()


plt.figure(figsize = (10,4))

plt.plot(frecuencias, PSD_X0_db, ls = ':', color= 'red', label  = f'f0 = {k0}hz', linewidth=1, marker='o', markersize=4)
plt.plot(frecuencias, PSD_X1_db, ls = '-.', color= 'blue', label  = f'f1 = {k1}hz', linewidth=1,        marker='o', markersize=4)
plt.plot(frecuencias, PSD_X2_db, ls = '-.', color= 'yellow', label  = f'f2 = {k2}hz', linewidth=1,      marker='o', markersize=4)


plt.title('Tercer Gráfica: Zoom de Desparramo espectral en potencia')
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.grid()
plt.xlim(240, 260)
plt.xticks(np.arange(240, 261, 1))

plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-10, 5)
plt.legend(loc='best')
plt.show()

# %% parte B identidad de parseval

#potencia en el dominio del tiempo:
#potencia media

pot0t= np.mean(x0**2) 
pot1t= np.mean(x1**2) 
pot2t= np.mean(x2**2) 

#potencia en el dominio de la frecuencia:
    
pot0f= np.sum(np.abs(np.fft.fft(x0)**2)) / N**2
pot1f= np.sum(np.abs(np.fft.fft(x1)**2)) / N**2
pot2f= np.sum(np.abs(np.fft.fft(x2)**2)) / N**2

print(f"\nSeñal 1: f0 = {f0}hz")
print(f'Potencia en dominio temporal: {pot0t}')
print(f'Potencia en dominio frecuencial: {pot0f}')


print(f"\nSeñal 2: f0 = {f1}hz")
print(f'Potencia en dominio temporal: {pot1t}')
print(f'Potencia en dominio frecuencial: {pot1f}')

print(f"\nSeñal 3: f0 = {f2}hz")
print(f'Potencia en dominio temporal: {pot2t}')
print(f'Potencia en dominio frecuencial: {pot2f}')


# %% parte C identidad de parseval

#forma manual
M = N + 9*N #10000
vector_pad = np.zeros(N*9)

x0_pad = np.append(x0, vector_pad) #vector de 10.000 espacios
#otra forma: np.concatenate((x0, vector_pad))
# otra forma: x0_padded = np.pad(x0, (0, M - N))


x1_pad = np.append(x1, vector_pad)
x2_pad = np.append(x2, vector_pad)


muestreo_pad = fs / M  # misma formula que delta_f pero conceptualmente es algo distinto

frecuencias_pad = np.arange(0, fs/2, fs/M) # de 0 a 500 con paso de 0,1 hz


# Otra forma mas directa era:fft_x0_pad = np.abs(np.fft.fft(x0, n=M)[:M // 2]) / N
fft_0_pad =np.abs( scipy.fft.fft(x0_pad)[:M//2])/N # se divide por N porque solo tengo 1000 valores distintos de 0
Max1 = np.max(fft_0 )
PSD_X0_pad   = (fft_0_pad/Max1)**2
PSD_X0_db_pad   = 10 * np.log10(PSD_X0_pad)



fft_1_pad =np.abs( scipy.fft.fft(x1_pad)[:M//2])/N # se divide por N porque solo tengo 1000 valores distintos de 0
PSD_X1_pad   = (fft_1_pad/Max1)**2
PSD_X1_db_pad   = 10 * np.log10(PSD_X1_pad)


fft_2_pad =np.abs( scipy.fft.fft(x2_pad)[:M//2])/N # se divide por N porque solo tengo 1000 valores distintos de 0
PSD_X2_pad   = (fft_2_pad/Max1)**2
PSD_X2_db_pad   = 10 * np.log10(PSD_X2_pad)


plt.figure(figsize = (12,5 ), dpi = 500)
plt.scatter(frecuencias_pad, PSD_X0_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250hz', marker='o', s=10)
plt.plot(frecuencias, PSD_X0_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250hz', marker='o', markersize=4, color = "orange")
plt.title('Densidad Espectral con Zero Padding')
plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.ylim(-80, 10)

plt.legend()
plt.show()


plt.figure(figsize = (12,5 ), dpi = 500)
plt.title('Zoom: Densidad Espectral con Zero Padding')
plt.plot(frecuencias_pad, PSD_X0_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250hz', marker='o', markersize=4)
plt.plot(frecuencias, PSD_X0_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250hz', marker='o', markersize=4)

plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlim(240,260)
plt.xticks(np.arange(240, 261, 1), rotation=90)
plt.tight_layout()

plt.legend()

plt.show()

# %%

plt.figure(figsize = (12,5 ), dpi = 500)
plt.scatter(frecuencias_pad, PSD_X1_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250.25hz', marker='o', s=10)
plt.plot(frecuencias, PSD_X1_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250.25hzz', marker='o', markersize=4, color = "orange")
plt.title('Densidad Espectral con Zero Padding')
plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.ylim(-80, 10)

plt.legend()
plt.show()


plt.figure(figsize = (12,5 ), dpi = 500)
plt.title('Zoom: Densidad Espectral con Zero Padding')
plt.plot(frecuencias_pad, PSD_X1_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250.25hz', marker='o', markersize=4)
plt.plot(frecuencias, PSD_X1_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250.25hz', marker='o', markersize=4)

plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlim(240,260)
plt.xticks(np.arange(240, 261, 1), rotation=90)
plt.ylim(-52, 10)

plt.tight_layout()

plt.legend()

plt.show()
# %%

plt.figure(figsize = (12,5 ), dpi = 500)
plt.scatter(frecuencias_pad, PSD_X2_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250.5hz', marker='o', s=10)
plt.plot(frecuencias, PSD_X2_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250.5hz', marker='o', markersize=4, color = "orange")
plt.title('Densidad Espectral con Zero Padding')
plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.ylim(-80, 10)

plt.legend()
plt.show()


plt.figure(figsize = (12,5 ), dpi = 500)
plt.title('Zoom: Densidad Espectral con Zero Padding')
plt.plot(frecuencias_pad, PSD_X2_db_pad, label = 'DFT con Zero Padding ko = 0, f0 = 250hz', marker='o', markersize=4)
plt.plot(frecuencias, PSD_X2_db, label = 'DFT sin Zero Padding ko = 0, f0 = 250hz', marker='o', markersize=4)

plt.grid()
plt.xlabel("Frecuencias [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlim(240,260)
plt.xticks(np.arange(240, 261, 1), rotation=90)
plt.ylim(-52, 10)

plt.tight_layout()

plt.legend()

plt.show()

