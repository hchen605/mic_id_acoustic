import numpy as np
import scipy.signal as ss
import soundfile as sf
#import rir_generator as rir
import os
import librosa


data_dir =[
#'/home/koredata/hsinhung/speech/C1_mid_full_1021/',
#'/home/koredata/hsinhung/speech/C2_clo_full_1030/',
#'/home/koredata/hsinhung/speech/C3_mid_full_1027/',
#'/home/koredata/hsinhung/speech/C4_mid_full_1025/',
#'/home/koredata/hsinhung/speech/D1_mid_full_1013/',
#'/home/koredata/hsinhung/speech/D2_mid_full_1028/',
#'/home/koredata/hsinhung/speech/D3_mid_full_1027/',
#'/home/koredata/hsinhung/speech/D4_mid_full_1023/',
#'/home/koredata/hsinhung/speech/D5_mid_full_1019/',
#'/home/koredata/hsinhung/speech/M1_mid_full_1022/',
#'/home/koredata/hsinhung/speech/M2_mid_full_1021/',
#'/home/koredata/hsinhung/speech/M3_mid_full_1009/',
'/home/koredata/hsinhung/speech/Mobile_Recording/iPhone',
'/home/koredata/hsinhung/speech/Mobile_Recording/MI',
'/home/koredata/hsinhung/speech/Mobile_Recording/LG',
'/home/koredata/hsinhung/speech/Mobile_Recording/Samsung',
'/home/koredata/hsinhung/speech/Mobile_Recording/Moto',
'/home/koredata/hsinhung/speech/Mobile_Recording/OPPO'
]

rir_dir = '/home/koredata/hsinhung/speech/rir_medium_3'
rir_wav = '/home/hsinhung/microphone_classification_extend/rir/RIR_room/medium/smard_7_8_3_2_1Y.wav'


#print(h.shape)              # (4096, 3)
#print(signal.shape)         # (11462, 2)

for path in data_dir:
    #print(path)
    for (dirpath, dirnames, filenames) in os.walk(path):
        #print(dirpath)
        if '/DR1' in dirpath:
            print(dirpath)
            for f in filenames:
                if not f.endswith(".WAV"):
                    continue
                
                signal, fs = sf.read(os.path.join(dirpath, f), always_2d=True)
                signal = librosa.to_mono(signal.T)

                dirp = dirpath.split('/')

                rir, sr_rir = librosa.load(rir_wav)
                rir = librosa.to_mono(rir.T)
                rir = librosa.resample(rir, sr_rir, fs)
                rir = rir / np.abs(rir).max()
                # Convolve signal with impulse responses
                signal = ss.convolve(signal, rir)
                #signal = signal / np.abs(signal).max()
                #print(signal.shape)         # (15557, 2, 3)
                write_dir = os.path.join(rir_dir, dirpath[31:])
                
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)

                write_path = os.path.join(write_dir, f) 
                sf.write(write_path, signal, fs)

        if 'clo' in dirpath:
            print(dirpath)
            for f in filenames:
                if not f.endswith(".wav"):
                    continue
                signal, fs = sf.read(os.path.join(dirpath, f), always_2d=True)
                signal = librosa.to_mono(signal.T)

                dirp = dirpath.split('/')

                rir, sr_rir = librosa.load(rir_wav)
                rir = librosa.to_mono(rir.T)
                rir = librosa.resample(rir, sr_rir, fs)
                rir = rir / np.abs(rir).max()
                # Convolve signal with impulse responses
                signal = ss.convolve(signal, rir)

                #print(signal.shape)         # (15557, 2, 3)
                write_dir = os.path.join(rir_dir, dirpath[31:])
                
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)

                write_path = os.path.join(write_dir, f) 
                sf.write(write_path, signal, fs)