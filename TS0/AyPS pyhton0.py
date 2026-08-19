# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:31:02 2026

@author: Gonza
"""

import matplotlib.pyplot as plt
import numpy as np





#%%definiciones

fs = 1000 #hz
N = 1000 #muestras


#%% funciones

def mi_funcion_sen(vmax, dc, ff, ph, nn=N, fs=fs):
    
    tt = np.arange(nn) / fs
    # Enteros desde 0 hasta N-1, luego a esos numeros los divide por fs
    # print(tt) para ver de forma mas comoda los tiempos 
    # tt: tiempos en segundos
    
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    # xx: valores de senial en voltios
    # vmax: amplitud (en voltios)
    # dc, direct current, valor medio de la senial (en voltios)
    # ff frecuencia de la senial, ciclos por segundo (en hertz)
    # ph fase (en rad)
    # nn numero de muestras, adimensional
    # fs frecuencia de muestras en muestras/s (hz)
    
    #recuerdo: Ts=1/fs, periodo de muestreo (en segundos)     
    
    #entonces con fs=1000 (muestras por segundo) y N = 1000 muestras
    # el tiempo que cubren las muestras es = 1 segundo (estrictamente 0,999ms) y el que veo que cubre el grafico.
    
    # si reduzco la frecuencia de muestreo, entonces vere mas ciclos, ya que el grafico cubrira mas tiempo
    # y por lo tanto mas periodos de la senial
    
    # y que ocurre si aumento la frecuencia de muestreo? 
    return tt, xx

#%% comienzo de mi script


t, x = mi_funcion_sen(vmax=2, dc=0, ff=1, ph=0, nn=N, fs=fs)

#Las llamo x,t para separar la llamada de la funcion de la funcion original.
# grafica el valor de la muestra(x) en tiempo t

plt.plot(t, x)
plt.title('Señal: senoidal' )
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.show()


#en que momento deja de ser una sinusoidal

#probar cerca y lejos de nyquist