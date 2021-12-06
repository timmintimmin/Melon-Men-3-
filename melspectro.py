import numpy as np
import math
import scipy as sp
from scipy.io.wavfile import read as wavread

def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2 ** (nbits - 1))

        # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return (samplerate, audio)

def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])

    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]

        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))]
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization

    return X


def Meltool(blockSize, fs, Filters, fMax):

    # Initialization
    f_min = 0
    f_max = min(fMax, fs/2)
    f_fft = np.linspace(0, fs/2, blockSize//2+1)
    H = np.zeros((Filters, f_fft.size))

    # Compute center band frequencies
    minformel = 1000 * math.log2(1 + (f_min) / 1000)
    maxformel = 1000 * math.log2(1 + (f_max)/1000)
    merge = np.linspace(minformel, maxformel, Filters+2)
    f_mel = 1000 * (2 ** (merge / 1000) - 1)
    f_l = f_mel[0:iNumFilters]
    f_c = f_mel[1:iNumFilters + 1]
    f_u = f_mel[2:iNumFilters + 2]

    afFilterMax = 2 / (f_u - f_l)

    # Compute the transfer functions
    for c in range(iNumFilters):
        H[c] = np.logical_and(f_fft > f_l[c], f_fft <= f_c[c]) * \
            afFilterMax[c] * (f_fft-f_l[c]) / (f_c[c]-f_l[c]) + \
            np.logical_and(f_fft > f_c[c], f_fft < f_u[c]) * \
            afFilterMax[c] * (f_u[c]-f_fft) / (f_u[c]-f_c[c])

    return H, f_c

def MelSpectrogram(x, fs, blockSize=4096, hopSize=2048, Filters=128, fMax=None):

    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)

    X = compute_spectrogram(xb)
    # Convert power spectrum to magnitude spectrum
    X = np.sqrt(X / 2)

    # Compute Mel filters
    H, f_c = Meltool(blockSize, fs, Filters, fMax)

    M = np.matmul(H, X)

    return M, f_c, t
#      M: Mel spectrum
#      f_c: Center frequencies of mel bands
#      t: Timestamps

