import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torchaudio.transforms import *
from hyperparams import hp

class Util:
    def __init__(self):
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.symbol_to_id = {
          s: i for i, s in enumerate(hp.symbols)
        }
        
        self.MelSpectrogram=MelSpectrogram(
            sample_rate=hp.sr,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            n_mels=hp.mel_freq
        ).to(self.device)
        
        self.inverse_transform = torchaudio.transforms.InverseMelScale(
            sample_rate=hp.sr,
            n_stft=hp.n_stft,
            n_mels=hp.mel_freq
        ).to(self.device)
        
        self.grifflim_transform = torchaudio.transforms.GriffinLim(
            n_fft=hp.n_fft,
            hop_length=hp.hop_length
        ).to(self.device)

    def text_to_seq(self,text):
        text = text.lower()
        seq = []
        for s in text:
            _id = self.symbol_to_id.get(s, None)
            if _id is not None:
                seq.append(_id)
        if len(seq)<hp.seq_length-1:
            for i in range(hp.seq_length-len(seq)-1):
                seq.append(0)
        seq.append(self.symbol_to_id["EOS"])

        return torch.IntTensor(seq)
    
    
    def inverse_mel_spec_to_wav(self,mel_spec):
        mel_spec=mel_spec*10
        mel_spec=torchaudio.functional.DB_to_amplitude(mel_spec,ref=1,power=1) 
        spectrogram = self.inverse_transform(mel_spec)
        pseudo_wav = self.grifflim_transform(spectrogram)
        return pseudo_wav
    
    
util=Util()

    