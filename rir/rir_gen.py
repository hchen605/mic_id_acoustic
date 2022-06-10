import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import os

#signal, fs = sf.read("/home/koredata/hsinhung/speech/C1_mid_full_1021/TRAIN/DR1/MCPM0/SA1.WAV", always_2d=True)
data_dir =[
'/home/koredata/hsinhung/speech/C1_mid_full_1021/',
'/home/koredata/hsinhung/speech/C2_clo_full_1030/',
'/home/koredata/hsinhung/speech/C3_mid_full_1027/',
'/home/koredata/hsinhung/speech/C4_mid_full_1025/',
'/home/koredata/hsinhung/speech/D1_mid_full_1013/',
'/home/koredata/hsinhung/speech/D2_mid_full_1028/',
'/home/koredata/hsinhung/speech/D3_mid_full_1027/',
'/home/koredata/hsinhung/speech/D4_mid_full_1023/',
'/home/koredata/hsinhung/speech/D5_mid_full_1019/',
'/home/koredata/hsinhung/speech/M1_mid_full_1022/',
'/home/koredata/hsinhung/speech/M2_mid_full_1021/',
'/home/koredata/hsinhung/speech/M3_mid_full_1009/'
]
rir_dir = '/home/koredata/hsinhung/speech/rir_11_9_7p5m'


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
                dirp = dirpath.split('/')

                h = rir.generate(
                    c=340,                  # Sound velocity (m/s)
                    fs=fs,                  # Sample frequency (samples/s)
                    r=[                     # Receiver position(s) [x y z] (m)
                        [9.5, 2, 1]
                        
                    ],
                    s=[2, 2, 1],          # Source position [x y z] (m)
                    L=[11, 9, 3],            # Room dimensions [x y z] (m)
                    reverberation_time=0.5, # Reverberation time (s)
                    nsample=4096,           # Number of output samples
                    )

                # Convolve 2-channel signal with 3 impulse responses
                signal = ss.convolve(h[:, None, :], signal[:, :, None])

                #print(signal.shape)         # (15557, 2, 3)
                write_dir = os.path.join(rir_dir, dirpath[31:])
                
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)

                write_path = os.path.join(write_dir, f) 
                sf.write(write_path, signal[:,:,0], fs)