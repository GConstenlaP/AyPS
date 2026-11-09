# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:08:27 2026

@author: tomas
"""
import numpy as np
from scipy import signal as signal

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

##################
## ECG con ruido
##################

fs=1000

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one = mat_struct['ecg_lead'].flatten() * 1.0
ecg_one_lead=ecg_one[5000:12000]
#ecg_one_lead=ecg_one
N = len(ecg_one_lead)

wp=30

ws=40

Gpass,Gstop=0.5,60

sos=signal.iirdesign(wp,ws,Gpass,Gstop,ftype='butter',fs=fs,output='sos')

# 1. Aplanar matriz y definir vector de tiempo
ecg_one_lead = ecg_one_lead.squeeze()
t = np.arange(N) / fs

ecg_filtrado = signal.sosfilt(sos, ecg_one_lead)
# Bloque para graficar superpuestas
plt.figure(figsize=(12, 5))

# Trazar ambas señales en el mismo gráfico
plt.plot(t, ecg_one_lead, label="ECG Original",color="blue")
plt.plot(t, ecg_filtrado, label="ECG Filtrado", color="orange", linewidth=1.5)

plt.title("Comparación: ECG Original vs. ECG Filtrado")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
  # Primeros 5 segundos
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

#frecuencia

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

f_ecg,Pxx=sig.welch(ecg_filtrado.flatten(),fs,window='blackmanharris',nperseg=len(ecg_one_lead)/2)
plt.figure()
plt.plot(f_ecg,10*np.log10(Pxx),alpha=0.7)

plt.grid(True)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('dB')
plt.title('Potencia espectral de Electro Cardiograma filtrado')
plt.legend()
plt.show()

#delta
delta_kronecker=np.zeros(7000)




#def calc resp
