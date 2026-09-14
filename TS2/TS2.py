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

# Calculo de Potencia:
# modulo_fft_X   = np.abs(scipy.fft.fft(X)[:N//2]) / N
# Max1 = np.max(modulo_fft_X)
# PSD_X   = (modulo_fft_X /Max1)**2
# PSD_X_db   = 10 * np.log10(PSD_X)


# A partir de Ahora graficare solo el rango de (0, fs/2)
frecuencias = np.arange(0, fs/2, fs/N)

# FFTs normalizadas acotadas a Nyquist (0 a fs/2)
modulo_fft_X   = np.abs(scipy.fft.fft(X)[:N//2]) / N
modulo_fft_XR  = np.abs(scipy.fft.fft(XR)[:N//2]) / N
modulo_fft_XRQ = np.abs(scipy.fft.fft(XRQ)[:N//2]) / N

#Normalizacion de Señal respecto a amplitud maxima
#Escalo para que el pico max de amplitud de la señal original coincida con el 0
Max1 = np.max(modulo_fft_X)

# calculo de Potencia 
#La potencia de una componente espectral es la amplitud en v, al cuadrado

PSD_X   = (modulo_fft_X /Max1)**2
PSD_XR  = (modulo_fft_XR /Max1)**2
PSD_XRQ = (modulo_fft_XRQ /Max1)**2

#Conversión a dB (10 * log10 para Potencia)
PSD_X_db   = 10 * np.log10(PSD_X)
PSD_XR_db  = 10 * np.log10(PSD_XR)
PSD_XRQ_db = 10 * np.log10(PSD_XRQ)


#El piso analogico R es el valor medio del ruido (analogico) gaussiano que se agrega a la señal  antes de pasar por el adc
# El piso Digital Eq es el valor medio de potencia del error de cuantizacion Eq = XRQ - XR

#Error de cuantización
eq = XRQ - XR
modulo_fft_eq_norm = np.abs(scipy.fft.fft(eq)[:N//2]) / N
PSD_eq    = (modulo_fft_eq_norm / Max1)**2
PSD_eq_db = 10 * np.log10(PSD_eq)


piso_analog  = 10 * np.log10(np.mean(PSD_XR[2:]))
piso_digital = 10 * np.log10(np.mean(PSD_eq[2:]))


plt.figure()
plt.plot(frecuencias, PSD_X_db, color ='black', alpha = 1, zorder = 5, label = 'PSD de Señal original')
plt.plot(frecuencias, PSD_XR_db, color = 'Red', alpha = 0.7, zorder = 4, label = 'PSD de Señal con Ruido')
plt.plot(frecuencias, PSD_XRQ_db, color = 'Blue', alpha = 0.7, zorder = 6, label = 'PSD de Señal con Ruido cuantizada')
plt.plot(frecuencias, PSD_eq_db , color = 'Yellow', alpha = 0.7, zorder = 4, label = 'PSD del Error de cuantizacion')

#lineas de piso de Ruido
plt.axhline(piso_analog, color='red', linestyle='--', label=f'piso analog. ({piso_analog:.1f} dB)', zorder = 8)
plt.axhline(piso_digital, color='darkcyan', linestyle=':', label=f'piso digital ({piso_digital:.1f} dB)', lw = 4,zorder = 8)

plt.title("Densidades de Potencia de Señales")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True, linestyle='--', alpha=0.5)
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
plt.scatter(lags, autocorr_norm, color= 'blue', s = 15)
plt.title('Autocorrelación del Ruido de Cuantización (eq)')
plt.xlabel('Lag [muestras]')
plt.ylabel('Autocorrelación Normalizada')
plt.grid(True)
plt.ylim(-0.2, 1.1)
plt.show()

autocorr_potencia = np.correlate(eq, eq, mode='full') / N

# La potencia medida corresponde al lag central
potencia_medida = autocorr_potencia[N - 1]
# Potencia teórica del ruido de cuantización 
potencia_teorica = (q**2) / 12

# Error relativo porcentual
error_relativo = np.abs(potencia_medida - potencia_teorica) / potencia_teorica * 100

print(f"Potencia medida: {potencia_medida:.6f}")
print(f"Potencia teórica: {potencia_teorica:.6f}")
print(f"Error relativo: {error_relativo:.2f}%")

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
################################################################################################################################################
#Punto B 4 bits con k= 0,1 y k=10

Prq = (q**2) / 12       
Pra0 = Prq * 0.1 # #Potencia de Ruido analogico. Equivale a PQ por una constante K de escala
Pra10 = Prq * 10

#Señal de Ruido Analogico que entra al ADC generado con su potencia media PRA
R_0 = gen_noise_normal( varianza = Pra0)
R_10 = gen_noise_normal( varianza = Pra10)

# Señal con Ruido
XR0 = X + R_0
XR10 = X + R_10

# Señal Con Ruido Cuantizada
#Proceso del ADC
XRQ0 = np.round(XR0 / q) * q
XRQ10 = np.round(XR10 / q) * q

# divide el valor de la señal en el paso, para obtener el entero mas cercano al "nivel de la señal" y multiplica por q para obteer en voltios, en la escala de la señal original.
# redondea a los niveles discretos de 4 bits, son 16 niveles

plt.figure(figsize=(13, 4), dpi=500)
plt.plot(T, X, color = 'red', label = 'Señal Analógica Pura', linewidth=3, zorder = 4)
plt.plot(T, XR0, color = 'cyan', label = 'Señal con Ruido (K = 0,1)', linewidth=3,zorder = 3)
plt.plot(T, XR, color = 'Blue', label = 'Señal con Ruido (K = 1)', linewidth=3, zorder = 2)
plt.plot(T, XR10, color = 'Black', label = 'Señal con Ruido (K = 10)', linewidth=3, zorder = 1)



# plt.plot(T, X, color = 'black', label = 'Señal original', linewidth=2)
# plt.plot(T, XRQ, color = 'cyan', label = 'Señal muestreada por ADC', linewidth=0.4, marker='o', markersize=1, markeredgecolor='blue', markeredgewidth=0.5)
#alpha es la opacidad, 
plt.title('Comparación de Señales con Ruido analógico')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.legend(loc='best') #loc='upper right', loc='best'

#plt.tight_layout()
#plt.savefig('grafico.png', dpi=600) /guardar grafico en escritorio. DPI: calidad tambien cambiando png por pdf o svg cambiamos el formato de archivo

plt.show()


# %% Punto B  Cuantizacion de k = 0,1

# Señal Con Ruido Cuantizada
#Proceso del ADC
XRQ0 = np.round(XR0 / q) * q

plt.figure(figsize=(13, 4), dpi=500)
plt.plot(T, XR0, color = 'red', label = 'Señal con Ruido (K = 0,1)', linewidth=3)
plt.plot(T, X, color = 'black', label = 'Señal original', linewidth=2)
plt.plot(T, XRQ0, color = 'cyan', label = 'Señal muestreada por ADC 4 bits', linewidth=0.4, marker='o', markersize=1, markeredgecolor='blue', markeredgewidth=0.5)
#alpha es la opacidad, 
plt.title('Señal de 4 Bits(K = 0,1)')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.legend(loc='best') #loc='upper right', loc='best'
#plt.tight_layout()
#plt.savefig('grafico.png', dpi=600) /guardar grafico en escritorio. DPI: calidad tambien cambiando png por pdf o svg cambiamos el formato de archivo

plt.show()


# %% Punto B  Cuantizacion de k = 10

# Señal Con Ruido Cuantizada
#Proceso del ADC
XRQ10 = np.round(XR10 / q) * q

plt.figure(figsize=(13, 4), dpi=500)
plt.plot(T, XR10, color = 'red', label = 'Señal con Ruido (K = 10)', linewidth=3)
plt.plot(T, X, color = 'black', label = 'Señal original', linewidth=2)
plt.plot(T, XRQ10, color = 'cyan', label = 'Señal muestreada por ADC 4 bits', linewidth=0.4, marker='o', markersize=2, markeredgecolor='blue', markeredgewidth=0.5)
#alpha es la opacidad, 
plt.title('Señal de 4 Bits(K = 10)')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.legend(loc='best') #loc='upper right', loc='best'
#plt.tight_layout()
#plt.savefig('grafico.png', dpi=600) /guardar grafico en escritorio. DPI: calidad tambien cambiando png por pdf o svg cambiamos el formato de archivo

plt.show()

# %%  Segunda Grafica

# FFTs normalizadas acotadas a Nyquist (0 a fs/2)
modulo_fft_XR0  = np.abs(scipy.fft.fft(XR0)[:N//2]) / N
modulo_fft_XRQ0 = np.abs(scipy.fft.fft(XRQ0)[:N//2]) / N

modulo_fft_XR10  = np.abs(scipy.fft.fft(XR10)[:N//2]) / N
modulo_fft_XRQ10 = np.abs(scipy.fft.fft(XRQ10)[:N//2]) / N



PSD_XR0  = (modulo_fft_XR0 /Max1)**2
PSD_XRQ0 = (modulo_fft_XRQ0 /Max1)**2

PSD_XR10  = (modulo_fft_XR10 /Max1)**2
PSD_XRQ10 = (modulo_fft_XRQ10 /Max1)**2



PSD_XR0_db  = 10 * np.log10(PSD_XR0)
PSD_XRQ0_db = 10 * np.log10(PSD_XRQ0)

PSD_XR10_db  = 10 * np.log10(PSD_XR10)
PSD_XRQ10_db = 10 * np.log10(PSD_XRQ10)



#El piso analogico R es el valor medio del ruido (analogico) gaussiano que se agrega a la señal  antes de pasar por el adc
#El piso Digital Eq es el valor medio de potencia del error de cuantizacion Eq = XRQ - XR

#Error de cuantización
# k=0,1
eq0 = XRQ0 - XR0
modulo_fft_eq_norm0 = np.abs(scipy.fft.fft(eq0)[:N//2]) / N
PSD_eq0    = (modulo_fft_eq_norm0 / Max1)**2
PSD_eq0_db = 10 * np.log10(PSD_eq0)

piso_analog0  = 10 * np.log10(np.mean(PSD_XR0[2:]))
piso_digital0 = 10 * np.log10(np.mean(PSD_eq0[2:]))

# k=10
eq10 = XRQ10 - XR10
modulo_fft_eq_norm10 = np.abs(scipy.fft.fft(eq10)[:N//2]) / N
PSD_eq10    = (modulo_fft_eq_norm10 / Max1)**2
PSD_eq10_db = 10 * np.log10(PSD_eq10)

piso_analog10  = 10 * np.log10(np.mean(PSD_XR10[2:]))
piso_digital10 = 10 * np.log10(np.mean(PSD_eq10[2:]))



# grafico k=0,1

plt.figure()
plt.plot(frecuencias, PSD_XR0_db, color = 'Red', alpha = 0.7, zorder = 4, label = 'PSD de Señal con Ruido')
plt.plot(frecuencias, PSD_XRQ0_db, color = 'Blue', alpha = 0.7, zorder = 6, label = 'PSD de Señal con Ruido cuantizada')
plt.plot(frecuencias, PSD_eq0_db , color = 'Yellow', alpha = 0.7, zorder = 4, label = 'PSD del Error de cuantizacion')

#lineas de piso de Ruido
plt.axhline(piso_analog0, color='red', linestyle='--', label=f'piso analog. ({piso_analog0:.1f} dB)', zorder = 8)
plt.axhline(piso_digital0, color='darkcyan', linestyle=':', label=f'piso digital ({piso_digital0:.1f} dB)', lw = 4,zorder = 8)

plt.title("Señal Muestreada por ADC 4 bits k = 0,1")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(-85, 10)
plt.legend(loc='best')
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

# grafica k=10

plt.figure()
plt.plot(frecuencias, PSD_XR10_db, color = 'Red', alpha = 0.7, zorder = 4, label = 'PSD de Señal con Ruido')
plt.plot(frecuencias, PSD_XRQ10_db, color = 'Blue', alpha = 0.7, zorder = 6, label = 'PSD de Señal con Ruido cuantizada')
plt.plot(frecuencias, PSD_eq10_db , color = 'Yellow', alpha = 0.7, zorder = 4, label = 'PSD del Error de cuantizacion')

#lineas de piso de Ruido
plt.axhline(piso_analog10, color='red', linestyle='--', label=f'piso analog. ({piso_analog10:.1f} dB)', zorder = 8)
plt.axhline(piso_digital10, color='darkcyan', linestyle=':', label=f'piso digital ({piso_digital10:.1f} dB)', lw = 4,zorder = 8)

plt.title("Señal Muestreada por ADC 4 bits k = 10")
plt.ylabel("Densidad de Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(-85, 10)
plt.legend(loc='best')
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)




# %% k=0,1

#Histograma del error
#Error de cuantizacion eq = XRQ - XR


plt.figure()
plt.hist(eq0, edgecolor='black')
plt.title('Histograma del ruido de cuantización')
plt.xlabel('Magnitud del error del Ruido de cuantizacion')
plt.ylabel('Cantidad de muestras')
plt.grid()


# %% Autocorrelacion 
autocorr0 = np.correlate(eq0, eq0, mode='full')
lags0 = np.arange(-N + 1, N)

# Normalización respecto al pico central (lag 0)
autocorr_norm0 = autocorr0 / np.max(autocorr0)

plt.figure(figsize=(9, 4))
plt.scatter(lags0, autocorr_norm0, color= 'blue', s = 15)
plt.title('Autocorrelación del Ruido de Cuantización (eq)')
plt.xlabel('Lag [muestras]')
plt.ylabel('Autocorrelación Normalizada')
plt.grid(True)
plt.ylim(-0.2, 1.1)
plt.show()

autocorr_potencia0 = np.correlate(eq0, eq0, mode='full') / N

# La potencia medida corresponde al lag central
potencia_medida0 = autocorr_potencia0[N - 1]
# Potencia teórica del ruido de cuantización 
potencia_teorica0 = (q**2) / 12

# Error relativo porcentual
error_relativo0 = np.abs(potencia_medida0 - potencia_teorica0) / potencia_teorica0 * 100

print(f"Potencia medida: {potencia_medida0:.6f}")
print(f"Potencia teórica: {potencia_teorica0:.6f}")
print(f"Error relativo: {error_relativo0:.2f}%")

# %% Test Kolmogorov Smirnov
from scipy.stats import kstest

resultado_ks0 = kstest(eq0, 'uniform', args=(a, ancho_intervalo))

print("K-S:", resultado_ks0.statistic) #Es el estadístico \(D\) de Kolmogorov-Smirnov y me devuelve la maxima separacion obtenida entre ambas funciones acumuladas
print("p-value:", resultado_ks0.pvalue)

alpha = 0.05
if resultado_ks0.pvalue > alpha:
    print("No se rechaza H0. El error se comporta como una distribución Uniforme [-q/2, q/2].")
else:
    print("Se rechaza H0. El error NO sigue una distribución uniforme.")
    

# %% k=10

#Histograma del error
#Error de cuantizacion eq = XRQ - XR


plt.figure()
plt.hist(eq10, edgecolor='black')
plt.title('Histograma del ruido de cuantización')
plt.xlabel('Magnitud del error del Ruido de cuantizacion')
plt.ylabel('Cantidad de muestras')
plt.grid()


# %% Autocorrelacion 
autocorr10 = np.correlate(eq10, eq10, mode='full')
lags10 = np.arange(-N + 1, N)

# Normalización respecto al pico central (lag 0)
autocorr_norm10 = autocorr10 / np.max(autocorr10)

plt.figure(figsize=(9, 4))
plt.scatter(lags10, autocorr_norm10, color= 'blue', s = 15)
plt.title('Autocorrelación del Ruido de Cuantización (eq)')
plt.xlabel('Lag [muestras]')
plt.ylabel('Autocorrelación Normalizada')
plt.grid(True)
plt.ylim(-0.2, 1.1)
plt.show()

autocorr_potencia10 = np.correlate(eq10, eq10, mode='full') / N

# La potencia medida corresponde al lag central
potencia_medida10 = autocorr_potencia10[N - 1]
# Potencia teórica del ruido de cuantización 
potencia_teorica10 = (q**2) / 12

# Error relativo porcentual
error_relativo10 = np.abs(potencia_medida10 - potencia_teorica10) / potencia_teorica10 * 100

print(f"Potencia medida: {potencia_medida10:.6f}")
print(f"Potencia teórica: {potencia_teorica10:.6f}")
print(f"Error relativo: {error_relativo10:.2f}%")

# %% Test Kolmogorov Smirnov
from scipy.stats import kstest

resultado_ks10 = kstest(eq10, 'uniform', args=(a, ancho_intervalo))

print("K-S:", resultado_ks10.statistic) #Es el estadístico \(D\) de Kolmogorov-Smirnov y me devuelve la maxima separacion obtenida entre ambas funciones acumuladas
print("p-value:", resultado_ks10.pvalue)

alpha = 0.05
if resultado_ks10.pvalue > alpha:
    print("No se rechaza H0. El error se comporta como una distribución Uniforme [-q/2, q/2].")
else:
    print("Se rechaza H0. El error NO sigue una distribución uniforme.")