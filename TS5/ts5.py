# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:15:48 2026

@author: Usuario
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy import signal as sig
import scipy.io as sio
from scipy.io.wavfile import write

import sounddevice as sd


import os

#en que carpeta estoy
#print(os.getcwd())

#agrego mi carpeta al directorio
os.chdir(r'C:\Users\Gonza\AyPS\Repositorio AyPS Git\TS5')

#importo lib de señales
import scipy.signal.windows as win

from scipy import signal

#%% FUNCIONES

# nperseg: cantidad de muestras de cada segmento, noverlap = solapamiento, nfft? me hace el zeropadding, scaling: density devuelve escala de la fft en densidad de potencia v2/hz y scaling='spectrum' devuelve en amplitud de potecnia, return_onesided: para mostrar frec positivas  o + y -.

#funcion para estimar con welch la densidad de potencia. Me dice cuando el anacho de banda del o5 por ciento de la señal ne potencia.
def calcular_psd_y_bw(x, fs, ventana, nperseg, porcentaje):

    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=ventana,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=None,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=-1,
        average='mean'
    )
    

    df = f[1] - f[0]

    potencia_acum = np.cumsum(Pxx) * df
    potencia_total = potencia_acum[-1]

    bw = f[np.where(potencia_acum >= porcentaje * potencia_total)[0][0]]

    return f, Pxx, bw


#%% ################### Lectura de ECG sin ruido ###################

fs_ecg = 1000 # Hz

ecg_one_lead = np.load('ecg_sin_ruido.npy')

#los tiempos: son el vector de las muestras divido la frecuencia
t_ecg = np.arange(len(ecg_one_lead)) / fs_ecg

#print(len(ecg_one_lead)) #tengo 30mil muestras
#la resolucion frecuencial, deltaf es = fs/nperseg, ya que nperseg son las muestras de cada bloque
# con nperseg mas chico tengo mas bloques, entonces tengo menos varianza pero menor resulcion espectral.



plt.figure()
plt.title('Señal 1: Electrocardiograma')
plt.grid()
plt.plot(t_ecg, ecg_one_lead)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.show()



# Grafico para bloques de 1000, 2000, etc muestras
nperseg_values = [1000,2000,8000]

for nperseg in nperseg_values:
# UTLIZIZO VENTANA HANN

# UTLIZIZO VENTANA HANN 
    f_ecg, Pxx_ecg, bw_ecg = calcular_psd_y_bw(ecg_one_lead, fs_ecg, ventana = 'hann', nperseg = nperseg, porcentaje = 0.98)
    print(f'Ancho de banda de 98% potencia de ECG con bloques de {nperseg} muestras : {bw_ecg:.3f} Hz')
    
    
    L = nperseg
    var_teorica = (9 * L / (16 * len(ecg_one_lead))) * (np.mean(Pxx_ecg)**2)
    print(f'Varianza teórica estimada del estimador: {var_teorica:.5f}\n')
    
    Pxx_norm = Pxx_ecg / np.max(Pxx_ecg) # PSD en dB
    Pxx_db = 10 * np.log10(Pxx_norm)
    plt.figure(figsize=(10, 4))
    plt.plot(f_ecg, Pxx_db)
    plt.xlim(0, 35)
    plt.xticks(np.arange(0, 35, 1), rotation=90)
    plt.xlabel('Frecuencia [Hz]')
    plt.axvline(bw_ecg, color='black', linestyle='dashed', linewidth=2, label='98% ancho de banda')
    plt.ylabel('PSD [dB]')
    plt.title(f'PSD ECG - Welch - bloques de {nperseg} muestras (Δf ≈ {fs_ecg/nperseg:.3f} Hz)')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()



#%% zeropad
# f_zero, Pxx_zero = signal.welch(
#     ecg_one_lead,
#     fs=fs_ecg,
#     window='hann',
#     nperseg=1000,
#     noverlap= 1000 // 2,
#     nfft=24000,
#     detrend='constant',
#     return_onesided=True,
#     scaling='density',
#     axis=-1,
#     average='mean'
# )
# Pxx_norm_zero = Pxx_zero  / np.max(Pxx_zero )
# # PSD en dB
# Pxx_db_zero = 10 * np.log10(Pxx_norm_zero )


# plt.figure(figsize=(10, 4))
# plt.plot(f_zero,  Pxx_db_zero)
# plt.xlim(0, 35)
# plt.xticks(np.arange(0, 35, 1), rotation=90)
# plt.xlabel('Frecuencia [Hz]')
# plt.axvline(bw_ecg, color='black', linestyle='dashed', linewidth=2, label='98% ancho de banda')
# plt.ylabel('PSD [dB]')
# plt.title(f'PSD ECG - Welch - nperseg={nperseg} (Δf ≈ {fs_ecg/nperseg:.3f} Hz)')
# plt.grid()
# plt.legend(loc='upper right')
# plt.show()
 
 
# En todas las gráficas se puede visualizar el maximo en 1 hz que se presume que es la frecuencia del latido, sin embargo a medida que se aumendan las muestras de los bloques debido al aumentod e la resolucion espectral se pueden apreciar con mayor precisio´n los picos de frecuencais que antes eran enmascarados
    
#%% ##################################### Lectura de pletismografía (PPG)  #####################################

fs_ppg = 400 # Hz

##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

N_ppg = len(ppg)
t_ppg = np.arange(N_ppg) / fs_ppg
# print(f'Cantidad de muestras: {N_ppg}') # 44919 muestras y 112.3 s de duracion

plt.figure(figsize=(12, 4))
plt.plot(t_ppg, ppg)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal PPG')
plt.grid()
plt.show()


# Como tengo n = 44919 y fs = 400hz

nperseg_values_ppg = [400, 800, 3200]
    
for nperseg1 in nperseg_values_ppg:
# UTLIZIZO VENTANA HANN
    f_ppg, Pxx_ppg, bw_ppg = calcular_psd_y_bw(ppg, fs_ppg, ventana = 'hann', nperseg =  nperseg1, porcentaje = 0.98)
    print(f'Ancho de banda de 98% potencia de ECG con bloques de {nperseg1} muestras : {bw_ppg:.3f} Hz')

    Pxx_ppg_norm = Pxx_ppg / np.max(Pxx_ppg)
    # PSD en dB
    Pxx_ppg_db = 10 * np.log10(Pxx_ppg_norm)


    plt.figure(figsize=(10, 4))
    plt.plot(f_ppg, Pxx_ppg_db)
    # plt.xlim(0, 35)
    # plt.xticks(np.arange(0, 35, 1), rotation=90)
    plt.xlabel('Frecuencia [Hz]')
    plt.xlim(0,50)
    plt.axvline(bw_ppg, color='black', linestyle='dashed', linewidth=2, label='98% ancho de banda')
    plt.ylabel('PSD [dB]')
    plt.title(f'PSD PPG - Welch - bloques de {nperseg1} muestras (Δf ≈ {fs_ppg/nperseg1:.3f} Hz)')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

# %% ##################################### Lectura audios#####################################

def calcular_psd_y_bw(x, fs, ventana, nperseg, porcentaje):

    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=ventana,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=None,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=-1,
        average='mean'
    )

    df = f[1] - f[0]

    # Cálculo de la potencia acumulada
    potencia_acum = np.cumsum(Pxx) * df
    potencia_total = potencia_acum[-1]

    # Porcentaje de potencia que queda fuera del intervalo
    porcentaje_fuera = 1 - porcentaje

    # Se distribuye el porcentaje fuera del intervalo
    # en partes iguales entre las frecuencias mínima y máxima
    porcentaje_min = porcentaje_fuera / 2
    porcentaje_max = 1 - porcentaje_fuera / 2

    # Frecuencia mínima que delimita el intervalo de potencia
    indice_min = np.where(
        potencia_acum >= porcentaje_min * potencia_total
    )[0][0]

    f_min = f[indice_min]

    # Frecuencia máxima que delimita el intervalo de potencia
    indice_max = np.where(
        potencia_acum >= porcentaje_max * potencia_total
    )[0][0]

    f_max = f[indice_max]

    # Ancho del intervalo de frecuencias que contiene
    # el porcentaje de potencia especificado
    bw = f_max - f_min

    return f, Pxx, f_min, f_max, bw
# %% 

fs_audio, audio1 = sio.wavfile.read('la cucaracha.wav')
fs_audio, audio2  = sio.wavfile.read('prueba psd.wav')
fs_audio, audio3 = sio.wavfile.read('silbido.wav')

#fs es 48khz
N_audio1 = len(audio1)
t_audio1 = np.arange(N_audio1) / fs_audio

print(f'Cantidad de muestras: {N_audio1}') # 144000


plt.figure(figsize=(12, 4))
plt.plot(t_audio1, audio1)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal Audio 1: La cucaracha')
plt.grid()
plt.show()


nperseg_values_audio1 = [1000, 24000, 48000]
    
for nperseg_audio1 in nperseg_values_audio1:
    f_audio1, Pxx_audio1, f_min_1, f_max_1, bw_audio1 = calcular_psd_y_bw(
        audio1, fs_audio, ventana='hann', nperseg=nperseg_audio1, porcentaje=0.98
    )
    print(f'Audio 1 - nperseg={nperseg_audio1}: BW98 = {bw_audio1:} Hz (Desde {f_min_1:} Hz hasta {f_max_1:} Hz)')

    Pxx_audio1_norm = Pxx_audio1 / np.max(Pxx_audio1)
    Pxx_audio1_db = 10 * np.log10(Pxx_audio1_norm)

    plt.figure(figsize=(10, 4))
    plt.plot(f_audio1, Pxx_audio1_db)
    plt.xlim(0, 3000)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.axvline(f_min_1, color='red', linestyle='dashed', linewidth=1.5, label=f'f_min ({f_min_1} Hz)')
    plt.axvline(f_max_1, color='black', linestyle='dashed', linewidth=1.5, label=f'f_max ({f_max_1} Hz)')
    plt.title(f'PSD Audio 1: "La cucaracha" - Welch - bloques de {nperseg_audio1} muestras (Δf ≈ {fs_audio/nperseg_audio1} Hz)')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


# %% ##################################### AUDIO 2 #####################################

N_audio2 = len(audio2)
t_audio2 = np.arange(N_audio2) / fs_audio

print(f'Audio 2 - Cantidad de muestras: {N_audio2}')
print(f'Audio 2 - Duración: {N_audio2/fs_audio} s')


# Señal temporal
plt.figure(figsize=(12, 4))
plt.plot(t_audio2, audio2)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal Audio 2: Prueba PSD')
plt.grid()
plt.show()


nperseg_values_audio2 = [1000, 24000, 48000]

nperseg_values_audio2 = [1000, 24000, 48000]

for nperseg_audio2 in nperseg_values_audio2:
    f_audio2, Pxx_audio2, f_min_2, f_max_2, bw_audio2 = calcular_psd_y_bw(
        audio2, fs_audio, ventana='hann', nperseg=nperseg_audio2, porcentaje=0.98
    )

    print(
        f'Audio 2 - nperseg={nperseg_audio2}: '
        f'Δf ≈ {fs_audio/nperseg_audio2:.3f} Hz, '
        f'BW98 = {bw_audio2:.3f} Hz (Desde {f_min_2:.1f} Hz hasta {f_max_2} Hz)'
    )

    Pxx_audio2_norm = Pxx_audio2 / np.max(Pxx_audio2)
    Pxx_audio2_db = 10 * np.log10(Pxx_audio2_norm)

    plt.figure(figsize=(10, 4))
    plt.plot(f_audio2, Pxx_audio2_db)
    plt.xlim(0, 3000)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.axvline(f_min_2, color='red', linestyle='dashed', linewidth=1.5, label=f'f_min ({f_min_2} Hz)')
    plt.axvline(f_max_2, color='black', linestyle='dashed', linewidth=1.5, label=f'f_max ({f_max_2} Hz)')
    plt.title(
        f'PSD Audio 2: "Prueba PSD" - Welch - '
        f'nperseg={nperseg_audio2} '
        f'(Δf ≈ {fs_audio/nperseg_audio2} Hz)'
    )
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
    
sd.play(audio2, fs_audio)

# %% audio 3

N_audio3 = len(audio3)
t_audio3 = np.arange(N_audio3) / fs_audio

plt.figure(figsize=(12, 4))
plt.plot(t_audio3, audio3)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal Audio 3: Silbido')
plt.grid()
plt.show()




# PSD mediante Welch
nperseg_values_audio3 = [1000, 24000, 48000]

for nperseg_audio3 in nperseg_values_audio3:
    f_audio3, Pxx_audio3, f_min_3, f_max_3, bw_audio3 = calcular_psd_y_bw(
        audio3, fs_audio, ventana='hann', nperseg=nperseg_audio3, porcentaje=0.98
    )

    print(
        f'Audio 3 - nperseg={nperseg_audio3}: '
        f'Δf ≈ {fs_audio/nperseg_audio3:} Hz, '
        f'BW98 = {bw_audio3} Hz (Desde {f_min_3} Hz hasta {f_max_3} Hz)'
    )

    Pxx_audio3_norm = Pxx_audio3 / np.max(Pxx_audio3)
    Pxx_audio3_db = 10 * np.log10(Pxx_audio3_norm)

    plt.figure(figsize=(10, 4))
    plt.plot(f_audio3, Pxx_audio3_db)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [dB]')
    plt.xlim(0, 8000)
    plt.axvline(f_min_3, color='red', linestyle='dashed', linewidth=1.5, label=f'f_min ({f_min_3} Hz)')
    plt.axvline(f_max_3, color='black', linestyle='dashed', linewidth=1.5, label=f'f_max ({f_max_3} Hz)')
    plt.title(
        f'PSD Audio 3: "Silbido" - Welch - '
        f'nperseg={nperseg_audio3} '
        f'(Δf ≈ {fs_audio/nperseg_audio3:} Hz)'
    )
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
    
# %%
print("--- Resumen de Anchos de Banda (98% de Potencia) ---")

print(f"Señal 1 - ECG (6 bloques de L =8000): {bw_ecg:.2f} Hz")
print(f"Señal 2 - PPG (bloques de L = 3200): {bw_ppg:.2f} Hz")
print(f"Señal 3 - Audio 1 'La cucaracha' (L = 48000): desde {f_min_1:.1f} Hz hasta {f_max_1:.1f} Hz -> Ancho de Banda = {bw_audio1:.2f} Hz")
print(f"Señal 4 - Audio 2 'Prueba PSD' (L = 48000): desde {f_min_2:.1f} Hz hasta {f_max_2:.1f} Hz ->  Ancho de Banda = {bw_audio2:.2f} Hz")
print(f"Señal 5 - Audio 3 'Silbido' (L = 48000): desde {f_min_3:.1f} Hz hasta {f_max_3:.1f} Hz ->  Ancho de Banda = {bw_audio3:.2f} Hz")
# graficar audios
# plt.figure()
# plt.plot(wav_data1)
# plt.plot(wav_data2)
# plt.plot(wav_data3)

# escuchar audio:
# sd.play(wav_data1, fs_audio)


# w_blackmanharris = win.blackmanharris(N) # .reshape(N, 1)
# w_flattop = win.flattop(N)
# w_triang = win.triang(N)


# f, Pxx_spec  = signal.welch(wav_data1, fs=fs_audio, window=w_blackmanharris, nperseg=1000, noverlap= 1000 // 2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
# plt.figure()
# plt.semilogy(f, np.sqrt(Pxx_spec))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Linear spectrum [V RMS]')
# plt.show()

# f, Pxx_den_blackman =signal.welch(wav_data1, fs=fs_audio, window=w_blackmanharris, nperseg=1000, noverlap= 1000//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
# f1, Pxx_den_flattop =signal.welch(wav_data1, fs=fs_audio, window=w_flattop, nperseg=1000, noverlap= 1000//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
# f2, Pxx_den_triang =signal.welch(wav_data1, fs=fs_audio, window=w_triang, nperseg=1000, noverlap= 1000//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
# # f3, Pxx_den_zeropadding1 =signal.welch(wav_data1, fs=fs_audio, window=w_blackmanharris, nperseg=1000, noverlap= 1000//2, nfft=100, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
# # f4, Pxx_den_zeropadding2 =signal.welch(wav_data1, fs=fs_audio, window=w_blackmanharris, nperseg=1000, noverlap= 1000//2, nfft=1000, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')




# plt.figure()
# plt.plot(f, Pxx_den_blackman)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()


# plt.figure()
# plt.plot(f1, Pxx_den_flattop)
# plt.plot(f2, Pxx_den_triang)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
