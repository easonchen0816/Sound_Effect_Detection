import os
import csv
import pickle
import tables
import resampy

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    return wav_data, sr

# data and save
dataset_path = ['/mnt/NAS/henry/cat_user/furbo3/meta_furbo3_catuser_0315-0528.csv']
save_path = ['/mnt/NAS/henry/cat_user/furbo3/meta_furbo3_catuser_0315-0528.h5']

for i in range(len(dataset_path)):
    # read csv data
    all_data = pd.read_csv(dataset_path[i])
    data = all_data['path']

    # spec parameter
    sample_rate = 8000
    n_mels = 64
    hop_length = 80
    n_fft = 200
    inpt_x = 500

    # frequency normalize
    freq_norm_channel = False
    if freq_norm_channel:
        print("Frequency channel normalize")
    if not freq_norm_channel:
        print("original preprocessing")

    # .h5 file
    name_x = save_path[i]
    fx = tables.open_file(name_x, mode='w')
    atom = tables.Float32Atom()
    array_x = fx.create_earray(fx.root, 'data', atom, (0,1,inpt_x,n_mels))

    for path in tqdm(data):
        # read wav file
        try:
            wav_data, sr = wav_read(path)
            assert wav_data.dtype == np.int16
        except:
            print(path)
            continue

        # normalize
        samples = wav_data / 32768.0

        # convert to mono
        if len(samples.shape) > 1:
            samples = mp.mean(samples, axis=1)
        
        # resample
        if sr != sample_rate:
            samples = resampy.resample(samples, sr, sample_rate)
        
        # transform to mel spec
        spec = librosa.feature.melspectrogram(samples, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        spec_db = librosa.power_to_db(spec).T
        spec_db = np.concatenate((spec_db, np.zeros((inpt_x - spec_db.shape[0], n_mels))), axis=0) if spec_db.shape[0] < inpt_x else spec_db[:inpt_x]
        
        # frequency normalization
        if freq_norm_channel:
            spec_db = (spec_db - spec_db.mean(axis=0)) / (spec_db.std(axis=0) + 1e-9)
        
        inpt = np.reshape(spec_db, (1,1,spec_db.shape[0], spec_db.shape[1]))
        array_x.append(inpt)
    print(array_x.shape)
    fx.close()
