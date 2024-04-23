import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils import util
from hyperparams import hp

class AudioData(Dataset):
    def __init__(self,indices,util):
        self.SAMPLE_RATE=hp.sr
        self.transformation=util.MelSpectrogram
        self.annotations=pd.read_csv(hp.csv_path)
        self.audio_dir=hp.wav_path
        self.indices=indices
        self.num_samples=hp.num_samples
        self.device=util.device
        
    def __len__(self):
        return len(self.indices)
    
    def _get_audio_text(self,idx):
        text=str(self.annotations['text_norm'][idx])
        return util.text_to_seq(text)
    
    def _get_audio_sample_path(self,idx):
        name=str(self.annotations['wav'][idx])
        return self.audio_dir+name+'.wav'
    
    def _mix_down_if_necessary(self,signal):
        if signal.shape[0]!=1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
        
    def _resample_if_necessary(self,signal, sample_rate):
        if sample_rate!=self.SAMPLE_RATE:
            resampler=torchaudio.transforms.Resample(sample_rate,self.SAMPLE_RATE).to(self.device)
            signal=resampler(signal)
        return signal
        
    def _cut_if_necessary(self,signal):
        if signal.shape[1]>self.num_samples:
            signal=signal[:,:self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self,signal):
        length_signal=signal.shape[1]
        if length_signal<self.num_samples:
            num_missing_samples=self.num_samples-length_signal
            last_dim_padding=(0,num_missing_samples)
            signal=F.pad(signal,last_dim_padding)
        return signal
    
    def _pow_to_db_mel_spec(self,mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
                    mel_spec,
                    multiplier = 10, 
                    amin = 1e-10, 
                    db_multiplier = 1, 
                    top_db = 100
                    )
        mel_spec = mel_spec/10
        return mel_spec
    
    def __getitem__(self,idx):
        idx=self.indices[idx]
        audio_sample_path=self._get_audio_sample_path(idx)
        text=self._get_audio_text(idx)
        
        signal, sample_rate=torchaudio.load(audio_sample_path)
        signal=signal.to(self.device)
        # signal shape -> (num_channels,samples)
        signal=self._resample_if_necessary(signal,sample_rate)
                                                  #this is for converting audio samples with
                                                  #different samplerate to a fix samplerate
        
        signal=self._mix_down_if_necessary(signal)#this is for converting audio samples with
                                                  # different channels to tone down to mono
        
        signal=self._cut_if_necessary(signal)  # this basically removes excess sample part
        
        signal=self._right_pad_if_necessary(signal) # this pads the data to increase samples
        
        # signal shape -> (num_channels,num_samples)
        signal=self.transformation(signal) # (num_channels,num_samples) -> (num_channels,mel_freq,samples)
        # signal shape -> (num_channels,mel_freq,samples )
        
        signal=self._pow_to_db_mel_spec(signal)
        signal=signal.squeeze(0).transpose(0,1) # removed no. of channel dim, (mel_freq,samples)
        return signal,text


if __name__=='__main__':
    df=pd.read_csv(hp.csv_path)
    unique_id=np.arange(0,len(df))
    threshold=int(unique_id.shape[0]*0.9)
    train_id=unique_id[:threshold]
    test_id=unique_id[threshold:]
   
    train_dl=DataLoader(AudioData(train_id,util),batch_size=8)
    val_dl=DataLoader(AudioData(test_id,util),batch_size=8)
    