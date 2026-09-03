# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:48:26 2026

@author: Gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# %% definiciones

N = 1000

f0 = 2000 #2khz

fs = 20000 # frecuencia de muestreo tal que tengo al menos 10 muestras por periodo, fs ≥ 10 · f₀

n = np.arange(N)

t = n / fs

delta_f = fs / N  # resolución espectral en hz = 20

# %% funciones

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(nn) / fs

    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

    return tt, xx


def graficar_primer_ciclo(t, x, f, titulo):
    muestras_por_ciclo = int(fs / f)

    plt.figure()
    plt.plot(t[:muestras_por_ciclo + 1], x[:muestras_por_ciclo + 1],'o-')
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid()
    plt.show()

def gen_noise_normal(media=0, varianza=0.1, nn=N):

    desv_est = np.sqrt(varianza)

    ruido_normal = np.random.normal(media, desv_est, nn)

    return ruido_normal

def gen_noise_uniforme(varianza=0.1, nn=N):

    a = np.sqrt(3 * varianza)

    ruido_uniforme = np.random.uniform(-a, a, nn)

    return ruido_uniforme

def mi_funcion_pulso_rectangular(vmax=1, ff=1, duty=0.5, nn=N, fs=fs):

    tt = np.arange(nn) / fs

    xx = vmax * scipy.signal.square(2 * np.pi * ff * tt, duty=duty)

    return tt, xx

# %% comienzo de mi script

#senial 1 de 2 watts

#potencia media: P = A²/2, y senial 1 debe tener 1W de potencia

A1 = np.sqrt(2)

tt, xx = mi_funcion_sen(vmax=A1, dc=0, ff=f0, ph=0, nn=N, fs=fs)

plt.figure(1)
plt.plot(tt, xx)
plt.title('Señal 1: senoidal de 1 Watt')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.show()

graficar_primer_ciclo(tt, xx, f0, 'Primer ciclo de Señal 1')

#transformada de senial 1
fft_1= scipy.fft.fft(xx)

#acoto FFt y frec.
modulo_1 = np.abs(fft_1[:N//2])

frecuencias = np.arange(N//2) * delta_f


plt.figure(2)
plt.plot(frecuencias,modulo_1)
plt.title("Modulo FFT de señal 1")
plt.ylabel("|X[k]|")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

# %%
#senial 2 de 2 watts desfasado np.pi/2

#potencia media: P = A²/2, y senial 2 debe tener 2W de potencia
A2 = np.sqrt(2*2)

tt2, xx2 = mi_funcion_sen(vmax=A2, dc=0, ff=f0,ph=np.pi/2, nn=N, fs=fs)

plt.figure(3)
plt.plot(tt2, xx2)
plt.title('Señal 2: senoidal de 2 Watt')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.show()

graficar_primer_ciclo(tt2, xx2, f0, 'Primer ciclo de Señal 2')


#transformada de senial 2
fft_2= scipy.fft.fft(xx2)

modulo_2 = np.abs(fft_2[:N//2])

frecuencias2 = np.arange(N//2) * delta_f


plt.figure(4)
plt.plot(frecuencias2,modulo_2)
plt.title("Modulo FFT de señal 2")
plt.ylabel("|X[k]|")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

# %% Senial 3: ruido de distribucion normal media = 0 y varianza = 0,1

ruido_normal = gen_noise_normal()

plt.figure(5)
plt.plot(t, ruido_normal)
plt.title("Señal 3: Ruido normal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

fft_ruido_normal = scipy.fft.fft(ruido_normal)

modulo_ruido_normal = np.abs(fft_ruido_normal[:N//2])

plt.figure(6)
plt.plot(frecuencias, modulo_ruido_normal)
plt.title("Modulo FFT de señal 3: Ruido normal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

# %% Senial 4: ruido de distribucion uniforme media = 0 y varianza = 0,1

ruido_uniforme = gen_noise_uniforme(varianza=0.1)

plt.figure(7)
plt.plot(t, ruido_uniforme)
plt.title("Señal 4: Ruido uniforme")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()


fft_ruido_uniforme = scipy.fft.fft(ruido_uniforme)

modulo_ruido_uniforme = np.abs(fft_ruido_uniforme[:N//2])

plt.figure(8)
plt.plot(frecuencias, modulo_ruido_uniforme)
plt.title("Modulo FFT de señal 4: Ruido uniforme")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

# %% Señal 5: pulso rectangular

# Potencia media: P = A² · D
# Potencia = 1 W
# ciclo de actividad o duty cycle D = 50% = 0.5

A5 = 1
duty5 = 0.5


tt5, xx5 = mi_funcion_pulso_rectangular(vmax=A5, ff=f0, duty=duty5, nn=N, fs=fs)

plt.figure(9)
plt.plot(tt5, xx5)
plt.title("Señal 5: Pulso rectangular de 1 Watt")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

graficar_primer_ciclo(tt5, xx5, f0, 'Primer ciclo de Señal 5')

# Transformada de señal 5
fft_5 = scipy.fft.fft(xx5)

modulo_5 = np.abs(fft_5[:N//2])

plt.figure(10)
plt.plot(frecuencias, modulo_5)
plt.title("Módulo FFT de señal 5: Pulso rectangular")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

# %% ANALISIS
# P = E{x²}
P1 = np.mean(xx**2)
P2 = np.mean(xx2**2)
P3 = np.mean(ruido_normal**2)
P4 = np.mean(ruido_uniforme**2)
P5 = np.mean(xx5**2)

print("Potencia señal 1:", P1)
print("Potencia señal 2:", P2)
print("Potencia señal 3:", P3)
print("Potencia señal 4:", P4)
print("Potencia señal 5:", P5)