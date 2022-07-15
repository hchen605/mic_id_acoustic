import librosa
import numpy as np
import pandas as pd
import soundfile as sound

#from tensorflow import keras

import glob
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool

sr = 16000 
duration = 3

# +
def _load_data(data):
    wav, y_3, y_16 = data
    #print(wav)
    stereo, fs = sound.read(wav)
    #print(stereo.shape)
    #assert stereo.shape[0] > 0
    if stereo.shape[0] == 0:
        return 0.5*np.ones(sr*duration), y_3, y_16
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
    
    #stereo = librosa.resample(stereo, sr, 8000)
    #stereo = librosa.resample(stereo, 8000, sr)
    #assert stereo.shape[0] > 16000
    #noise only
    #orig_shape = stereo.shape
    #assert orig_shape[0] > 16000
    #trimmed, index = librosa.effects.trim(stereo, top_db=20)
    #stereo = np.concatenate((stereo[0:index[0]], stereo[index[1]:-1]))

    if stereo.shape[0] > sr*duration:
        #start = (stereo.shape[0] - sr*duration) // 2
        start = 1
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    return x, y_3, y_16

def load_data(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    datas = zip(wavpath, labels_3, labels_16)

    with Pool(32) as p:
        return p.map(_load_data, datas)
    
def _load_data_mel(data):
    wav, y_3, y_16 = data
    stereo, fs = sound.read(wav)
    #print(stereo.shape)
    #assert stereo.shape[0] > 0
    if stereo.shape[0] == 0:
        return 0.5*np.ones(sr*duration), y_3, y_16
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)

    if stereo.shape[0] > sr*duration:
        #start = (stereo.shape[0] - sr*duration) // 2
        start = 1
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    x = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, n_mels=128,
                                    fmax=8000)
    x = np.transpose(x,(1,0))#94,128
    

    return x, y_3, y_16

def load_data_mel(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    datas = zip(wavpath, labels_3, labels_16)

    with Pool(32) as p:
        return p.map(_load_data_mel, datas)

def _load_data_rir(data):
    wav, y_3, y_16, rir = data
    #print(wav)
    stereo, fs = sound.read(wav)
    #print(stereo.shape)
    assert stereo.shape[0] > 0
    
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
        
    #stereo = librosa.resample(stereo, sr, 8000)
    #stereo = librosa.resample(stereo, 8000, sr)
    #assert stereo.shape[0] > 16000
    #noise only
    #orig_shape = stereo.shape
    #assert orig_shape[0] > 16000
    #trimmed, index = librosa.effects.trim(stereo, top_db=20)
    #stereo = np.concatenate((stereo[0:index[0]], stereo[index[1]:-1]))
    if stereo.shape[0] > sr*duration:
        start = (stereo.shape[0] - sr*duration) // 2
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    return x, y_3, y_16, rir

def load_data_rir(data_csv, rir):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    #rir
    with open(rir) as f:
        feats = f.readlines()

    #rir
    rir = []
    for feat in feats:
        #count += 1
        #print(f'line {count}: {feat}') 
        x = feat.split(',')
        x = map(float,x)
        x = list(x)
        #print(x)
        rir = rir + [x]
        #print(data)    

    #rir = np.array(rir)

    #rir[:,0] = rir[:,0]/rir[:,0].max()
    #rir[:,1] = rir[:,1]/rir[:,1].max()
    #rir= list(rir)

    datas = zip(wavpath, labels_3, labels_16, rir)

    with Pool(32) as p:
        return p.map(_load_data_rir, datas)




