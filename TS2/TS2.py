# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:51:33 2026

@author: Gonza
"""



# %% Librerias y Parametros

import numpy as np
import matplotlib.pyplot as plt
import scipy


N = 1000            #muestras
fs = 1000           #frecuencia de muestreo del ADC


f0 = fs/N           #frecuencia señal FUNCION DE FS Y N 
delta_f = fs / N    #f0 = fs/N = delta_f, densidad espectral
    
n = np.arange(N)
# t = n / fs



# %% funciones

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):

    tt = np.arange(nn) / fs

    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)

    return tt, xx
 
def gen_noise_normal(media=0, varianza=0.1, nn=N):

    desv_est = np.sqrt(varianza) #sigma

    ruido_normal = np.random.normal(media, desv_est, nn)
#$$ R \sim \mathcal{N}(\mu,\sigma^2) $$
    return ruido_normal




# def graficar_primer_ciclo(t, x, f, titulo, ejex = 'Tiempo [s]', ejey = 'Amplitud [V]'):
    
#     muestras_por_ciclo = int(fs / f)
    
#     plt.figure()
#     plt.plot(t[:muestras_por_ciclo + 1], x[:muestras_por_ciclo + 1],'o-')
#     plt.title(titulo)
#     plt.xlabel(ejex)
#     plt.ylabel(ejey)
#     plt.grid()
#     plt.show()

# Aplicacion: graficar_primer_ciclo(t, XR, f0, titulo='Primer ciclo de Señal Con Ruido de Cuantizacion', ejex = 'A' , ejey = 'B')


# %% Señal 1

# Señal que entra al ADC:  XR=  X + R : señal original + ruido analogico

#Señal
#energia normalizada, varianza unitaria, por lo tanto la amplitud = raiz de 2
A1 = np.sqrt(2)
T, X = mi_funcion_sen(vmax=A1, dc=0, ff=f0, ph=0, nn=N, fs=fs)


# R, ruido analogico:  será incorrelado y Gaussiano. Uso distribucion normal
# El ADC que desemos simular trabajará a una frecuencia de muestreo fS=1000 Hz y tendrá un rango analógico de ±VF=2 Volts.

Vf = 2
#paso de cuantizacion q, con B bits = 4 es q = 0,25V
B = 4
q = (2*Vf) / (2**B)         # q paso de cuantizacion: q = 2*Vf / 2**B........ q(Resolucion del ADC o Paso de cuantizacion)= Rango total/ Niveles



#La potencia del ruido analogico será Pra = k * Prq. Donde k es una constante que escala el ruido Pq definido tomando de referencia de medicion el error de cuantizacion. Por definicion potencia del error de redondeo por una constante K

Prq = (q**2) / 12       
K = 1
Pra = Prq * K # #Potencia de Ruido analogico. Equivale a PQ por una constante K de escala

#Señal de Ruido Analogico que entra al ADC generado con su potencia media PRA
R = gen_noise_normal(varianza=Pra)





# Señal con Ruido
XR = X + R

# Señal Con Ruido Cuantizada
#Proceso del ADC
XRQ = np.round(XR / q) * q
# divide el valor de la señal en el paso, para obtener el entero mas cercano al "nivel de la señal" y multiplica por q para obteer en voltios, en la escala de la señal original.
# redondea a los niveles discretos de 4 bits, son 16 niveles




plt.figure(figsize=(13, 4), dpi=500)
plt.plot(T, XR, color = 'red', label = 'Señal con Ruido', linewidth=3)
plt.plot(T, X, color = 'black', label = 'Señal original', linewidth=2)
plt.plot(T, XRQ, color = 'cyan', label = 'Señal muestreada por ADC', linewidth=0.4, marker='o', markersize=1, markeredgecolor='blue', markeredgewidth=0.5)
#alpha es la opacidad, 
plt.title('Señal Con Ruido de Cuantizacion')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.legend(loc='best') #loc='upper right', loc='best'

#plt.tight_layout()
#plt.savefig('grafico.png', dpi=600) /guardar grafico en escritorio. DPI: calidad tambien cambiando png por pdf o svg cambiamos el formato de archivo

plt.show()


# error de cuantizacion : eQ = XQ - X
#Histograma del error
# %%  Segunda Grafica


#FFTs normalizadas por N
fft_X= scipy.fft.fft(X)
modulo_X_norm = np.abs(fft_X)/N


fft_XR= scipy.fft.fft(XR)
modulo_XR_norm = np.abs(fft_XR)/N


fft_XRQ= scipy.fft.fft(XRQ)
modulo_XRQ_norm = np.abs(fft_XRQ)/N



# A partir de Ahora graficare solo el rango de (0, fs/2)
frecuencias = np.arange(0, fs/2, fs/N)
frecuencias = frecuencias[:N//2]

#Escalo para que el pico max de amplitud de la señal original coincida con el 0
#Normalizacion de Señal respecto a amplitud maxima
Max1 = np.max(modulo_X_norm)

#modulo de fft normalizado con maximo
PSD_X = modulo_X_norm/Max1
PSD_XR = modulo_XR_norm/Max1
PSD_XRQ = modulo_XRQ_norm/Max1


# Magnitud espectral relativa en db
PSD_X_db = 20*np.log10((modulo_X_norm/Max1) + 1e-12)
PSD_XR_db = 20*np.log10((modulo_XR_norm/Max1) + 1e-12)
PSD_XRQ_db = 20*np.log10((modulo_XRQ_norm/Max1) + 1e-12)

#El piso analogico ñ es el valor medio del ruido (analogico) gaussiano que se agrega a la señal  antes de pasar por el adc
# El piso Digital ÑQ es el valor medio de potencia del error de cuantizacion Eq = XRQ - XR


#Error de cuantizacion
eq = XRQ - XR
fft_eq = scipy.fft.fft(eq) / N
modulo_eq_norm = np.abs(fft_eq[:N//2])
PSD_eq_db = 20 * np.log10((modulo_eq_norm  / Max1) + 1e-12)

piso_analog  = np.mean(PSD_XR_db[2:])
piso_digital = np.mean(PSD_eq_db)

plt.figure()
plt.plot(frecuencias, PSD_X_db[:N//2], color ='Blue', alpha = 0.5, zorder = 2, label = 'PSD de Señal original')
plt.plot(frecuencias, PSD_XR_db[:N//2], color = 'Red', alpha = 0.5, zorder = 4, label = 'PSD de Señal con Ruido')
plt.plot(frecuencias, PSD_XRQ_db[:N//2], color = 'Cyan', alpha = 0.9, zorder = 3, label = 'PSD de Señal con Ruido cuantizada')

#lineas de piso de Ruido
plt.axhline(piso_analog, color='red', linestyle='-', label=f'piso analog. ({piso_analog:.1f} dB)')
plt.axhline(piso_digital, color='darkcyan', linestyle='-', label=f'piso digital ({piso_digital:.1f} dB)')

plt.title("Densidades de Potencia de Señales")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid()
plt.ylim(-85, 10)
plt.legend(loc='best')
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)


# %% Error del ruido de cuantizacion
# Histograma del error
#Error de cuantizacion eq = XRQ - XR


plt.figure()
plt.hist(eq, edgecolor='black')
plt.title('Histograma del ruido de cuantización')
plt.xlabel('Magnitud del error del Ruido de cuantizacion')
plt.ylabel('Cantidad de muestras')
plt.grid()

#El histograma es una forma de representar la probabilidad experimental del error de cuantizacion que introduce el ADC, viendo el grafico podemos ver si se aproxima a una variable aleatoria de distribucion uniforme o si se aleja, pero no es suficiente para determinar.
# Para una distribucion uniforme ideal, todas las barras deberian tener la misma altura, es decir deberia haber la misma cantidad de muestras en cada valor de amplitud de error,


#el modelo teorico espera: que nq este entre -q/2 y q/2

# reviso minimo y maximo de nq, no deberia encontrar valoresmuy lejos de ±qq/2 
print("mínimo:", np.min(eq)) #en V
print("máximo:", np.max(eq))
print("qq/2 es:",eq/2)


# Con Q = 0,25, el rango del erri es -q/2 y q/2 = [-0,125, 0,125]

#Varianza

var_real = np.var(eq)
var_teorica = (q**2) / 12
print("El paso de cuantizacion es:", q) # V
print("Varianza experimental:", var_real)
print("Varianza teórica:", var_teorica)




# %% Autocorrelacion 
autocorr = np.correlate(eq, eq, mode='full')
lags = np.arange(-N + 1, N)

# Normalización respecto al pico central (lag 0)
autocorr_norm = autocorr / np.max(autocorr)

plt.figure(figsize=(9, 4))
plt.plot(lags, autocorr_norm, color='navy', linewidth=1)
plt.title('Autocorrelación del Ruido de Cuantización ($e_q$)')
plt.xlabel('Retardo / Lag [muestras]')
plt.ylabel('Autocorrelación Normalizada')
plt.grid(True)
plt.ylim(-0.2, 1.1)
plt.show()


# %% Test Kolmogorov Smirnov
from scipy.stats import kstest

a = -q / 2
b = q / 2
ancho_intervalo = b - a # = q

resultado_ks = kstest(eq, 'uniform', args=(a, ancho_intervalo))

print("K-S:", resultado_ks.statistic) #Es el estadístico \(D\) de Kolmogorov-Smirnov y me devuelve la maxima separacion obtenida entre ambas funciones acumuladas
print("p-value:", resultado_ks.pvalue)

alpha = 0.05
if resultado_ks.pvalue > alpha:
    print("No se rechaza H0. El error se comporta como una distribución Uniforme [-q/2, q/2].")
else:
    print("Se rechaza H0. El error NO sigue una distribución uniforme.")
    

# %% Punto B 
#Punto A fue con : K =  1, B = 4


#correlación y frecuencia y mo se relaciona con la cantidad de muestras
# como se calcula el ruido de cuantizacion  o error de ruyido de cuantizacion, 
#snrq = 6bit que es?

#relacion potencia de señal cn potencia de rudo
#log 10(psen/pruido)


