import torch
import os
import torch.nn as nn
import numpy as np
import torchaudio
import pandas as pd
from tqdm import tqdm
from hyperparams import hp
from model import SimpleTransformer
from utils import util
from torch.utils.data import DataLoader,Dataset
from dataset import AudioData

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cwd=os.getcwd()

df=pd.read_csv(hp.csv_path)
unique_id=np.arange(0,len(df))
threshold=int(unique_id.shape[0]*0.9)
train_id=unique_id[:threshold]
test_id=unique_id[threshold:]

train_dl=DataLoader(AudioData(train_id,util),batch_size=hp.batch_size)
val_dl=DataLoader(AudioData(test_id,util),batch_size=hp.batch_size)

model=SimpleTransformer().to(device)

if  os.path.exists(os.path.join(cwd,hp.save_name)):
    model.load_state_dict(torch.load(os.path.join(cwd,hp.save_name),map_location=device))
    print('Model Weights Loaded')
else:
    print('Model weights not loaded')

optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
criterion=nn.MSELoss()
model.train()
for epoch in range(hp.num_epochs):
    i=0
    total_loss=0
    for mel,text in tqdm(train_dl):
        if len(text)<hp.batch_size:
            continue
        optimizer.zero_grad()
        mel,text=mel.to(device),text.to(device)
        mel_postnet,mel_linear=model(mel,text)
        loss=criterion(mel_postnet,mel)+criterion(mel_linear,mel)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        i+=1
        
        if i%100==0:
            print(loss)
            torch.save(model.state_dict(),os.path.join(cwd,hp.save_name))
    print('Epoch Loss:',total_loss)

text = util.text_to_seq("Hello, world.").unsqueeze(0).to(device)
mel_postnet=model.inference(text)
pseudo_wav=util.inverse_mel_spec_to_wav(mel_postnet.transpose(1,2)[0])
torchaudio.save(os.path.join(cwd,'audio.wav'), pseudo_wav.unsqueeze(0).detach().cpu(), hp.sr,format="wav")
    

