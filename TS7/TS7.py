# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:15:11 2026

@author: tomas
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from matplotlib.patches import Rectangle
# %%
##################
## ECG con ruido
##################

fs=1000

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']*1.0
ecg_one=ecg_one_lead[5000:12000].flatten()
#ecg_one_lead=ecg_one
N = len(ecg_one_lead)
y_min, y_max = -100, 5

#%%
#############
# Plantilla #
#############
 

nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 2 # Hz
wp1 = 5 # Hz
wp2 = 25 # Hz
ws2 = 35 # Hz
wp=[wp1, wp2]
ws=[ws1, ws2]


 #%%
 #PARTE B
 #Para hallar la frecuencia de paso, wp1 y wp2, las mismas se fijaron 5 y 25 Hz como margenes interiores, respectivamente. En la TS5, se determinó que el ancho de banda de la señal ECG que se está trabajando es de unos 22 hz, por lo que con este rango se asegura atrapar todo el contendio espectral
 # Para hallar la frecuencia de rechazo, ws1 y ws2, se establecieron como márgenes exteriores 2 y 35 Hz, respectivamente. El motivo de estas selecciones es que, a baja frecuencia, el movimiento de la linea base inducido por la respiración genera ruido que se busca atenuar. Mismamente, luego de los 35 hz también hay ruido, del tipo muscular y por movimientos del electrodo
 # El ripple en la banda de paso se estableció en 1 dB ya que es la máxima variación que quiero permitirle a mi señal dentro de la banda de paso
 #La atenuación en la banda de rechazo se estableció en 40 dB para asegurarme de que las fuentes de ruido mencionadas queden atenuadas y no afecten la señal de interés.
#%%

#Parte C
sos_butt = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='butter', fs=fs, output = 'sos')
sos_cheby2 = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='cheby2', fs = fs, output = 'sos')
sos_cauer = sig.iirdesign(wp, ws, gpass=ripple, gstop=atenuacion, ftype='cauer', fs = fs, output = 'sos')
def calc_resp(sos, leg_str, fig_n_axes):

    # Respuesta en frecuencia en Hz, de 0 a Nyquist
    f, h = sig.freqz_sos(sos, worN=8192, fs=fs)

    # Módulo en dB y fase desenrollada en radianes
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))

    # Retardo de grupo: -d(fase)/d(omega), en segundos
    omega = 2 * np.pi * f
    retardo_s = -np.gradient(fase_rad, omega)

    # Donde el filtro rechaza la señal, el retardo no es significativo
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_s_plot = np.where(valido, retardo_s, np.nan)

    (fig, (ax_mod, ax_fase)) = fig_n_axes

    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.5, label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)
    ax_mod.legend()

    # --- Fase ---
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)
    for w in wp:
        ax_fase.axvline(w, color="gray", linestyle=":")
    for w in ws:
        ax_fase.axvline(w, color="gray", linestyle="--")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)
    ax_fase.legend()

    # --- Retardo de grupo (figura 2) ---
    plt.figure(2)
    plt.plot(f, retardo_s_plot, label=leg_str)
    for w in wp:
        plt.axvline(w, color="orange", linestyle="--")
    for w in ws:
        plt.axvline(w, color="red", linestyle=":")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (s)")
    plt.title("Retardo de grupo del filtro IIR")
    plt.grid(True)
    plt.legend()

#%% Graficar la respuesta de los filtros IIR

plt.close('all')

fig_n_axes = plt.subplots(
    2, 1, sharex=True, num=1, figsize=(10, 7), constrained_layout=True
)

calc_resp(sos_butt,   'butter', fig_n_axes)
calc_resp(sos_cheby2, 'cheby2', fig_n_axes)
calc_resp(sos_cauer,  'cauer',  fig_n_axes)

# Plantilla sobre el gráfico de módulo
ax_mod = fig_n_axes[1][0]

# Zona prohibida en rechazo: de -atenuacion hacia arriba
ax_mod.add_patch(Rectangle((0, -atenuacion), ws1, y_max + atenuacion,
                           facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))
ax_mod.add_patch(Rectangle((ws2, -atenuacion), nyq_frec - ws2, y_max + atenuacion,
                           facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))

# Franja prohibida bajo el ripple, en la banda de paso
ax_mod.add_patch(Rectangle((wp1, y_min), wp2 - wp1, -ripple - y_min,
                           facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx",
                           label='Zonas Plantilla'))

# Límites de las bandas
for w in wp:
    ax_mod.axvline(w, color="blue", linestyle=":", alpha=0.6)
for w in ws:
    ax_mod.axvline(w, color="red", linestyle="--", alpha=0.6)

ax_mod.legend(loc="lower left", fontsize=8)
ax_mod.set_xlim(0, 100)          
# Retardo de grupo (figura 2)
plt.figure(2)
plt.xlim(0, 100)                 
plt.show()         
#%% Paso la señal por los IIR
ecg_one_1d=ecg_one.ravel()

ecg_filt_butt=sig.sosfiltfilt(sos_butt,ecg_one_1d)
ecg_filt_cheby2=sig.sosfiltfilt(sos_cheby2,ecg_one_1d)
ecg_filt_cauer=sig.sosfiltfilt(sos_cauer,ecg_one_1d)
#%% Función para comparar la señal original y las señales IIR filtradas juntas
def plotear_todas_iir_juntas(ecg_orig, dic_filtradas_iir):
    plt.figure(figsize=(14, 6))
    # Graficamos la señal original con ruido de fondo
    plt.plot(ecg_orig, label='ECG con ruido', color='gray', alpha=0.5, linewidth=1)
    
    colores = ['blue', 'green', 'purple']
    
    for (nombre, ecg_filt), color in zip(dic_filtradas_iir.items(), colores):
        # Como usamos sosfiltfilt (fase cero), no hay retardo que compensar
        plt.plot(ecg_filt, label=f'IIR ({nombre})', color=color, linewidth=1.1, alpha=0.85)
        
    plt.title("Comparación de señales ECG filtradas con filtros IIR")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
#%% Señal pasada por los IIR
# Diccionario con las señales IIR filtradas
dic_iir_filtradas = {
    'Butterworth': ecg_filt_butt,
    'Cheby2': ecg_filt_cheby2,
    'Cauer': ecg_filt_cauer
}

# --- Llamamos a la función ---
plotear_todas_iir_juntas(ecg_one, dic_iir_filtradas)

#%%ARMADO DE FIRS
numtaps_hamming = 1501  
numtaps_blackman = 1501  
numtaps_hann = 1501  
numtaps_boxcar = 2001

# plantilla normalizada a Nyquist en dB
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, 0, 0, -atenuacion, -atenuacion])#disminuyo el máximo ripple permitido a 0 dB así puedo asegurarme de que no se atenue de mas en los límites de la banda de paso

# convertimos a veces para las funciones de diseño
gains = 10**(gains/20)

coefs_fir_hamming = sig.firwin2(numtaps_hamming, frecs, gains,window='hamming')
coefs_fir_blackman = sig.firwin2(numtaps_blackman, frecs, gains,window='blackman')
coefs_fir_hann = sig.firwin2(numtaps_hann, frecs, gains,window='hann')
coefs_fir_boxcar = sig.firwin2(numtaps_boxcar, frecs, gains,window='boxcar')

def respuesta_frec (coefs):
    f, h = sig.freqz(coefs, worN=8192, fs=fs)
    return f,h

f_fir_hamming, h_fir_hamming = respuesta_frec(coefs_fir_hamming)
f_fir_blackman, h_fir_blackman = respuesta_frec(coefs_fir_blackman)
f_fir_hann, h_fir_hann = respuesta_frec(coefs_fir_hann)
f_fir_boxcar, h_fir_boxcar = respuesta_frec(coefs_fir_boxcar)
#%% Función para graficar la respuesta de los filtros FIR
def calc_resp_fir(coefs, leg_str, fig_n_axes):
    # Respuesta en frecuencia con sig.freqz
    f, h = sig.freqz(coefs, worN=8192, fs=fs)

    # Módulo en dB y fase desenrollada en radianes
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))

    # Retardo de grupo (para los FIR simétricos es una constante hermosa)
    omega = 2 * np.pi * f
    retardo_s = -np.gradient(fase_rad, omega)

    (fig, (ax_mod, ax_fase)) = fig_n_axes

    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.5, label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla - FIR")
    ax_mod.grid(True)
    ax_mod.legend(loc="lower left", fontsize=8)

    # --- Fase ---
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)
    for w in wp:
        ax_fase.axvline(w, color="gray", linestyle=":")
    for w in ws:
        ax_fase.axvline(w, color="gray", linestyle="--")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)
    ax_fase.legend(loc="lower left", fontsize=8)

    plt.show()

    # Figura aparte para el retardo de grupo del FIR
    plt.figure(figsize=(10, 4))
    plt.plot(f, retardo_s, label=leg_str)
    for w in wp:
        plt.axvline(w, color="orange", linestyle="--")
    for w in ws:
        plt.axvline(w, color="red", linestyle=":")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (s)")
    plt.title("Retardo de grupo del filtro FIR")
    plt.grid(True)
    plt.legend()
#%% Graficar todos los FIR
plt.close('all')

fig_fir_axes = plt.subplots(
    2, 1, sharex=True, figsize=(10, 7), constrained_layout=True
)

calc_resp_fir(coefs_fir_hamming, 'Hamming', fig_fir_axes)
calc_resp_fir(coefs_fir_blackman, 'Blackman', fig_fir_axes)
calc_resp_fir(coefs_fir_hann, 'Hann', fig_fir_axes)
calc_resp_fir(coefs_fir_boxcar, 'Boxcar (Rectangular)', fig_fir_axes)
#Superponer la plantilla en el gráfico de módulo de los FIR
ax_mod_fir = fig_fir_axes[1][0]

# 1. Zona prohibida inferior (de 0 a ws1, por debajo de -atenuacion)
ax_mod_fir.add_patch(Rectangle((0, y_min), ws1, -atenuacion - y_min,
                               facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))

# 2. Zona prohibida superior (de ws2 hasta nyq_frec, por debajo de -atenuacion)
ax_mod_fir.add_patch(Rectangle((ws2, y_min), nyq_frec - ws2, -atenuacion - y_min,
                               facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))

# 3. Franja central prohibida por debajo del ripple (en la banda de paso entre wp1 y wp2)
ax_mod_fir.add_patch(Rectangle((wp1, y_min), wp2 - wp1, -ripple - y_min,
                               facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx", label='Zonas Plantilla'))

# Líneas verticales para los márgenes de las bandas
for w in wp:
    ax_mod_fir.axvline(w, color="blue", linestyle=":", alpha=0.6)
for w in ws:
    ax_mod_fir.axvline(w, color="red", linestyle="--", alpha=0.6)

ax_mod_fir.legend(loc="lower left", fontsize=8)
plt.draw()
#%% Filtrado de señal

ecg_filt_hamming = sig.lfilter(coefs_fir_hamming, 1.0, ecg_one)
ecg_fir_blackman = sig.lfilter(coefs_fir_blackman, 1.0, ecg_one)
ecg_fir_hann     = sig.lfilter(coefs_fir_hann,     1.0, ecg_one)
ecg_fir_boxcar   = sig.lfilter(coefs_fir_boxcar,   1.0, ecg_one)
#%% Función para comparar la señal original y las cuatro señales filtradas juntas
def ecg_filtrado_FIR(ecg_orig, filtradas_dict, numtaps_dict):
    plt.figure(figsize=(14, 6))
    # Graficamos la señal original de fondo (tomando como referencia el retardo máximo o medio)
    plt.plot(ecg_orig, label='ECG con ruido', color='gray', alpha=0.5, linewidth=1)
    
    colores = ['blue', 'green', 'orange', 'purple','red','yellow']
    
    for (nombre, ecg_filt), numtaps, color in zip(filtradas_dict.items(), numtaps_dict.values(), colores):
        retardo = (numtaps - 1) // 2
        ecg_filt_alineada = ecg_filt[retardo:]
        plt.plot(ecg_filt_alineada, label=f'Filtro {nombre}', color=color, linewidth=1.1, alpha=0.85)
        
    plt.title("Comparación de señales ECG filtradas con diferentes ventanas FIR")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Diccionarios para ordenar los datos y pasárselos a la función
dic_filtradas = {
    'Hamming': ecg_filt_hamming,
    'Blackman': ecg_fir_blackman,
    'Hann': ecg_fir_hann,
    'Boxcar': ecg_fir_boxcar
}

dic_taps = {
    'Hamming': numtaps_hamming,
    'Blackman': numtaps_blackman,
    'Hann': numtaps_hann,
    'Boxcar': numtaps_boxcar
}

#%% --- Llamamos a la función ---
ecg_filtrado_FIR(ecg_one, dic_filtradas, dic_taps)
#%% ================Método de cuadrados mínimos para diseño de FIR=========================
#=========================================================================================
desired = np.array([0.0,0.0,1.0,1.0,0.0,0.0])
numtaps_firls = 301

coefs_firls = sig.firls(numtaps_firls, frecs*nyq_frec, desired,fs=fs)
#%% Graficar respuesta de Firls
fig_firls_axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
calc_resp_fir(coefs_firls, 'Least-Squares (firls)', fig_firls_axes)

# Superponer la plantilla en el módulo
ax_mod_firls = fig_firls_axes[1][0]
ax_mod_firls.add_patch(Rectangle((0, y_min), ws1, -atenuacion - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))
ax_mod_firls.add_patch(Rectangle((ws2, y_min), nyq_frec - ws2, -atenuacion - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))
ax_mod_firls.add_patch(Rectangle((wp1, y_min), wp2 - wp1, -ripple - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx", label='Zonas Plantilla'))
for w in wp:
    ax_mod_firls.axvline(w, color="blue", linestyle=":", alpha=0.6)
for w in ws:
    ax_mod_firls.axvline(w, color="red", linestyle="--", alpha=0.6)
ax_mod_firls.legend(loc="lower left", fontsize=8)
plt.draw()
#%% Filtrado de señal con cuadrados minimmos
ecg_filt_firls = sig.lfilter(coefs_firls, 1.0, ecg_one)

#%% 2. Agregar firls al diccionario de los FIR para plotearlo junto con los demás
dic_filtradas['Least-Squares'] = ecg_filt_firls
dic_taps['Least-Squares'] = numtaps_firls

# Volvemos a llamar a la función para ver a firls compitiendo con Hamming, Blackman, Hann y Boxcar
ecg_filtrado_FIR(ecg_one, dic_filtradas, dic_taps)
#%% ================Método de Remez para diseño de FIR=========================
#=========================================================================================
numtaps_remez = 501  
bands_remez = np.array([0.0, ws1, wp1, wp2, ws2, nyq_frec])
desired_remez = np.array([0.0, 1.0, 0.0])  #(Rechazo, Paso, Rechazo)
weight_remez=np.array([3.0,1.0,3.0])#Le estoy diciendo que le de más peso de error a las bandas de rechazo para que cumpla con la plantilla, es decir, que sea menos tolerante con el error 

coefs_remez = sig.remez(numtaps_remez, bands_remez, desired_remez,weight=weight_remez, fs=fs)

#%% Graficar respuesta en frecuencia de Remez
fig_remez_axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
calc_resp_fir(coefs_remez, 'Equiripple (Remez)', fig_remez_axes)

# Superponer la plantilla en el módulo
ax_mod_remez = fig_remez_axes[1][0]
ax_mod_remez.add_patch(Rectangle((0, y_min), ws1, -atenuacion - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))
ax_mod_remez.add_patch(Rectangle((ws2, y_min), nyq_frec - ws2, -atenuacion - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx"))
ax_mod_remez.add_patch(Rectangle((wp1, y_min), wp2 - wp1, -ripple - y_min, facecolor="red", alpha=0.15, edgecolor="none", hatch="xxx", label='Zonas Plantilla'))
for w in wp:
    ax_mod_remez.axvline(w, color="blue", linestyle=":", alpha=0.6)
for w in ws:
    ax_mod_remez.axvline(w, color="red", linestyle="--", alpha=0.6)
ax_mod_remez.legend(loc="lower left", fontsize=8)
plt.draw()

#%% Filtrado de señal con Remez y actualización de gráficos temporales
ecg_filt_remez = sig.lfilter(coefs_remez, 1.0, ecg_one)

# Agregamos Remez a nuestros diccionarios de FIRs
dic_filtradas['Remez'] = ecg_filt_remez
dic_taps['Remez'] = numtaps_remez

# Llamamos a la función unificada para ver todas las señales FIR superpuestas (incluyendo Remez)
ecg_filtrado_FIR(ecg_one, dic_filtradas, dic_taps)