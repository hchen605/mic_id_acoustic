import os
import warnings
import multiprocessing as mp

import tqdm
import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional


class MicClassification(td.Dataset):
    def __init__(self,
                train_csv: str,
                dev_csv: str,
                label_type: str,
                train: bool = True,
                sample_rate: int = 44100,
                transform=None,
                target_transform=None,
                limit=np.inf):
        super(MicClassification, self).__init__()
        self.root = train_csv if train else dev_csv
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.sample_rate = sample_rate
        self.limit = limit if train else np.inf

        self.label_to_index = dict()
        self.data = dict()
        self.label_set = None
        self.load_data()
        self.indices = list(self.data.keys())
        

        
        self.class_idx_to_label = dict()
        for i, _ in enumerate(self.label_set):
            self.class_idx_to_label[i] = self.label_set[i]
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        
    @staticmethod
    def _load_worker(idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        if not os.path.exists(filename):
            print("Warning: file %s does not exist, skipped..." % filename, file=sys.stderr)
            return -1,-1,np.zeros(0)

        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
        if wav.ndim == 1: wav = wav[:, np.newaxis]
        wav = wav.T * 32768
        return idx, sample_rate, wav.astype(np.float32)
    def load_data(self):
        meta = pd.read_csv(self.root, sep='\t')
        assert self.label_type in meta, \
            "Error: label_type %s isn't found. Need to be in %s" % (self.label_type, list(meta))
        label_set = sorted(set(meta[self.label_type].values))
        self.label_set = label_set #['C1','C2','C3','C4','D1','D2','D3','D4','...','P3','P4','P5','P6']
        self.label_to_index = {label:idx for idx, label in enumerate(label_set)}

        label_count = {label:0 for label in label_set}
        items_to_load = []
        for idx, row in meta.iterrows():
            label = row[self.label_type]
            if label_count[label] >= self.limit: continue
            label_count[label] += 1
            items_to_load.append((idx, row['filename'], self.sample_rate))
        
        warnings.filterwarnings("ignore")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                if len(wav) == 0: continue
                
                row = meta.loc[idx]
                #target = self.label_to_index[row[self.label_type]]
                target = row[self.label_type]

                self.data[idx] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': target
                }

    def __getitem__(self, index) -> Tuple[np.ndarray, List[str]]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: str = self.data[self.indices[index]]['target']
        
        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)
       
        #print(audio)
        return audio, [target]


    def __len__(self) -> int:
        return len(self.data)
