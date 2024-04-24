import torch.nn as nn
import torch
import math
from tqdm import tqdm
import module
from hyperparams import hp
import torch.nn.functional as F


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class InputEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings=nn.Embedding(num_embeddings=hp.text_num_embeddings,embedding_dim=hp.encoder_embedding_size)
        
        self.linear1=nn.Linear(hp.encoder_embedding_size,hp.encoder_embedding_size)
        self.linear2=nn.Linear(hp.encoder_embedding_size,hp.embedding_size)
        
        self.conv1=nn.Conv1d(
            hp.encoder_embedding_size,hp.encoder_embedding_size,
            kernel_size=hp.encoder_kernel_size,stride=1,padding=1
                             )
        self.conv2=nn.Conv1d(
            hp.encoder_embedding_size,hp.encoder_embedding_size,
            kernel_size=hp.encoder_kernel_size,stride=1,padding=1
                             )
        self.bn1=nn.BatchNorm1d(hp.encoder_embedding_size)
        self.bn2=nn.BatchNorm1d(hp.encoder_embedding_size)
        
        self.dropout1=nn.Dropout(hp.dropout)
        self.dropout2=nn.Dropout(hp.dropout)
        
    def forward(self,x):
        x=self.embeddings(x)  # batch,seq_len -> batch,seq_len,encoder_embedding_size
        x=self.linear1(x)     # batch,seq_len,encoder_embedding_size -> batch,seq_len,encoder_embedding_size
        x=x.transpose(1,2)    # batch,seq_len,encoder_embedding_size -> batch,encoder_embedding_size,seq_len
        x=F.relu(self.bn1(self.conv1(x)))  # batch,encoder_embedding_size,seq_len -> batch,encoder_embedding_size,seq_len
        x=self.dropout1(x)
        x=F.relu(self.bn2(self.conv2(x)))  # batch,encoder_embedding_size,seq_len -> batch,encoder_embedding_size,seq_len
        x=self.dropout2(x)
        x=x.transpose(1,2)    # batch,encoder_embedding_size,seq_len -> batch,seq_len,encoder_embedding_size
        x=self.linear2(x)     # batch,seq_len,encoder_embedding_size -> batch,seq_len,embedding_size
        return x
        
        
class PositionalEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.positionalembedding=nn.Embedding(hp.max_mel_time,hp.embedding_size)
    def forward(self,x):
        x1=self.positionalembedding(torch.arange(hp.max_mel_time).to(device))
        return x+x1[:x.shape[1]]
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(
            embed_dim=hp.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
            )
    def forward(self,x,z=None):
        if z != None:
            return self.attn(query=x,key=z,value=z)
        return self.attn(query=x,key=x,value=x)
    

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.normlayer1=nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.normlayer2=nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.attn=MultiHeadAttentionBlock()
        self.linear1=nn.Linear(hp.embedding_size,hp.dim_feedforward)
        self.linear2=nn.Linear(hp.dim_feedforward,hp.embedding_size)
        self.dropout=nn.Dropout(hp.dropout)
        self.dropout1=nn.Dropout(hp.dropout)
        self.dropout2=nn.Dropout(hp.dropout)
    
    def forward(self,x):
        x_out=self.normlayer1(x)
        x_out,_=self.attn(x_out)
        x_out=self.dropout(x_out)
        x=x+x_out
        x_out=self.normlayer2(x)
        x_out=self.dropout1(self.linear1(x_out))
        x_out=self.dropout2(self.linear2(x_out))
        x=x+x_out
        
        return x
  
    
class PreProcess_Mel_Spec(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess=nn.Sequential(
            nn.Linear(hp.mel_freq,hp.embedding_size),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.embedding_size,hp.embedding_size),
            nn.ReLU(),
            nn.Dropout(hp.dropout)
            )
        self.positional_embeddings=PositionalEmbeddings()

    def forward(self,x):
        mel_x=self.preprocess(x)
        return self.positional_embeddings(mel_x)
    

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.normlayer1=nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.normlayer2=nn.LayerNorm(normalized_shape=hp.embedding_size)
        self.self_attn=MultiHeadAttentionBlock()
        self.attn=MultiHeadAttentionBlock()
        self.linear1=nn.Linear(hp.embedding_size,hp.dim_feedforward)
        self.linear2=nn.Linear(hp.dim_feedforward,hp.embedding_size)
        self.dropout1=nn.Dropout(hp.dropout)
        self.dropout2=nn.Dropout(hp.dropout)
        self.dropout3=nn.Dropout(hp.dropout)
        self.dropout4=nn.Dropout(hp.dropout)
        
    def forward(self,x,memory=None):
        x_out,_=self.self_attn(x)
        x_out=self.dropout1(x_out)
        x=x+x_out
        x=self.normlayer1(x)
        x_out,_=self.attn(x,memory)
        x_out=self.dropout2(x_out)
        x=x+x_out
        x=self.normlayer2(x)
        x_out=self.dropout3(self.linear1(x))
        x_out=self.dropout4(self.linear2(x_out))
        x=x+x_out
        return x
        

class PostNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(hp.embedding_size,hp.mel_freq) # batch,time,embedding_size -> batch,time,mel_freq
        # now we need to transpose the time and mel_freq, for the conv1d to operate
        self.conv1=nn.Conv1d(hp.mel_freq,hp.postnet_embedding_size,kernel_size=hp.encoder_kernel_size,stride=1,padding=1)
        self.bn1=nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout1=nn.Dropout(hp.dropout)
        
        self.conv2=nn.Conv1d(hp.postnet_embedding_size,hp.postnet_embedding_size,kernel_size=hp.encoder_kernel_size,stride=1,padding=1)
        self.bn2=nn.BatchNorm1d(hp.postnet_embedding_size)
        self.dropout2=nn.Dropout(hp.dropout)
        
        self.conv3=nn.Conv1d(hp.postnet_embedding_size,hp.mel_freq,kernel_size=hp.encoder_kernel_size,stride=1,padding=1)
        self.bn3=nn.BatchNorm1d(hp.mel_freq)
        self.dropout3=nn.Dropout(hp.dropout)
        
    def forward(self,x): # batch,time,embedding_size
        mel_linear=self.linear(x) # batch,time,embedding_size -> batch,time,mel_freq
        mel_linear=mel_linear.transpose(1,2) # batch,time,mel_freq -> batch,mel_freq,time
        
        mel_postnet=self.dropout1(F.tanh(self.bn1(self.conv1(mel_linear))))
        mel_postnet=self.dropout2(F.tanh(self.bn2(self.conv2(mel_postnet))))
        mel_postnet=self.dropout3(F.tanh(self.bn3(self.conv3(mel_postnet))))
        
        mel_postnet=mel_postnet.transpose(1,2) # batch,mel_freq,time -> batch,time,mel_freq
        mel_linear=mel_linear.transpose(1,2) # batch,mel_freq,time -> batch,time,mel_freq
        mel_postnet=mel_postnet+mel_linear
        
        return mel_postnet,mel_linear


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings=InputEmbeddings()
        self.positional_embeddings=PositionalEmbeddings()
        
        self.encoder_block1=EncoderBlock()
        self.encoder_block2=EncoderBlock()
        self.encoder_block3=EncoderBlock()
        
        # norm_memory
        
        self.preprocess_mel_spec=PreProcess_Mel_Spec()
        
        self.decoder_block1=DecoderBlock()
        self.decoder_block2=DecoderBlock()
        self.decoder_block3=DecoderBlock()
        
        self.postnet=PostNet()
        
    def forward(self,mel,text):
        text_out=self.input_embeddings(text)
        text_out=self.positional_embeddings(text_out)
        
        text_out=self.encoder_block1(text_out)
        text_out=self.encoder_block2(text_out)
        text_out=self.encoder_block3(text_out)
        # norm memory
        mel_out=self.preprocess_mel_spec(mel)
        
        mel_out=self.decoder_block1(mel_out,text_out)
        mel_out=self.decoder_block2(mel_out,text_out)
        mel_out=self.decoder_block3(mel_out,text_out)
        
        mel_postnet,mel_linear=self.postnet(mel_out)
        
        return mel_postnet,mel_linear
        
    @torch.no_grad()
    def inference(self,text,max_length=862):
        self.eval()
        N=text.shape[0]
        mel_fake=torch.zeros(N,1,hp.mel_freq,device=device) # shape is batch=1 time=1 and mel_freq since this 
                                                            # fake mel will append rest timestamp data with itself
        for _ in tqdm(range(max_length)):
            mel_postnet, mel_linear = self(mel_fake, text)
            mel_fake=torch.cat((mel_fake,mel_postnet[:,-1:,:]),dim=1)
            
        return mel_postnet
        
        
        
