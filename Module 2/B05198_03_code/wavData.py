# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:47:15 2015

@author: dj
"""

import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

sig, samplerate = sf.read('data/sound.wav')
sig.shape
slice=sig[0:500,:]
ft=np.abs(np.fft.fft(slice))

plt.plot(ft)
plt.plot(slice)