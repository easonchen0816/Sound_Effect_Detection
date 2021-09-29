import os
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import resampy
import librosa

from torch.utils.data import Dataset


try:
    def wav_read(params, wav_file, type_data):
        wav_file = os.path.join(params.dvc_root, type_data ,wav_file)
        wav_data, sr = sf.read(wav_file, dtype='int16')
        return wav_data, sr

except ImportError:
    def wav_read(wav_file):
        raise NotImplementedError('WAV file reading requires soundfile package.')


class SoundDataset(Dataset):
    """Create Sound Dataset with loading wav files and labels from csv.

    Attributes:
        params: A class containing all the parameters.
        data_type: A string indicating train or val.
        csvfile: A string containing our wav files and labels.
        normalize: A boolean indicating spectrogram is normalized to -1 to 1 or not.
        mixup: A boolean indicating whether to do mixup augmentation or not.
        preload: A boolean indicating whether to preload the spectrogram into memory or not.
    """
    def __init__(self, params, train=True):
        """Init SoundDataset with params
        Args:
            params (class): all arguments parsed from argparse
            train (bool): train or val dataset
        """
        self.params = params
        self.data_type = "train" if train else "val"
        self.csvfile = os.path.join(params.csv_root, "{}_{}.csv".format(self.params.name_prefix, self.data_type))
        self.preload = self.params.preload

        self.X, self.Y, self.filenames = self.read_data(self.csvfile)
        if self.preload:
            self.X = self.convert_to_spec(self.X)
            self.shape = self.get_shape(self.X[0])
        else:
            self.shape = self.get_shape(self.preprocessing(self.X[0][0], self.X[0][1]))

    def read_data(self, csvfile):
        """Read wav file from csv
        Args:
            csvfile: A string specifying the path of csvfile
        Return:
            data: A list of tuple (wav data in np.int16 data type, sampling rate of wav file)
            label: A list of labels corresponding to the wav data
            filenames: A list of filenames of the wav file
        """
        df = pd.read_csv(csvfile, error_bad_lines=False)
        data, label, filenames = [], [], []
        print("reading wav files...")
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            wav_data, sr = wav_read(self.params, row.path, self.data_type)
            assert wav_data.dtype == np.int16
            data.append((wav_data, sr))
            label.append(row.label)
            filenames.append(row.path)
        return data, label, filenames

    def convert_to_spec(self, data):
        """Convert wav_data into log mel spectrogram.
        Args:
            data: A list of tuple (wav data in np.int16 data type, sampling rate of wav file)
        Return:
            A list of log mel spectrogram
        """
        print("convert to log mel spectrogram...")
        return [self.preprocessing(wav, sr) for wav, sr in tqdm(data)]
    
    def get_shape(self, x):
        """Get the shape of input data.
        """
        return x.shape

    def preprocessing(self, wav_data, sr):
        """Convert wav_data to log mel spectrogram.
            1. normalize the wav_data
            2. convert the wav_data into mono-channel
            3. resample the wav_data to the sampling rate we want
            4. compute the log mel spetrogram with librosa function
        Args:
            wav_data: An np.array indicating wav data in np.int16 datatype
            sr: An integer specifying the sampling rate of this wav data
        Return:
            inpt: An np.array indicating the log mel spectrogram of data
        """
        # normalize wav_data 16 bits data
        samples = wav_data / float(self.params.normalize_num)

        # convert samples to mono-channel file
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)

        # resample samples to 8k
        if sr != self.params.sr:
            samples = resampy.resample(samples, sr, self.params.sr)

        # transform samples to mel spectrogram
        inpt_x = self.params.inp
        spec = librosa.feature.melspectrogram(y=samples, sr=self.params.sr, n_fft=self.params.nfft, hop_length=self.params.hop, n_mels=self.params.mel)
        spec_db = librosa.power_to_db(spec).T
        spec_db = np.concatenate((spec_db, np.zeros((inpt_x - spec_db.shape[0], self.params.mel))), axis=0) if spec_db.shape[0] < inpt_x else spec_db[:inpt_x]
        inpt = np.reshape(spec_db, (1, spec_db.shape[0], spec_db.shape[1]))

        return inpt.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = self.X[idx] if self.params.preload else self.preprocessing(self.X[idx][0], self.X[idx][1])
        label = self.Y[idx]

        return spec.astype('float32'), label
