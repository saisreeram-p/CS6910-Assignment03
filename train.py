# Importing Modules
import wandb
import torch 
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn  import functional
from pytorch_lightning.loggers import WandbLogger
from matplotlib.font_manager import FontProperties
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import random
import csv
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class Encoder(nn.Module):
  ''' This is the model class used to define the Encoder structure and to train the neural network on given parameters'''
  def __init__(self,input_size,embedding_size,hidden_size,layers,cell_type,bidirectional,dropout):
    '''
      input_size: The size of the input vocabulary or the number of distinct tokens.
      embedding_size: The size of the embedding vectors for each token.
      hidden_size: The size of the hidden state of the RNN cell.
      layers: The number of layers in the RNN.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      bidirectional: A boolean flag indicating whether the RNN is bidirectional or not.
      dropout: The dropout probability for the RNN layers
    '''
    super(Encoder,self).__init__()
    self.cell_type = cell_type
    self.embedding = nn.Embedding(input_size,embedding_size)
    self.rnn = cell_type(embedding_size,hidden_size,layers,bidirectional = bidirectional,dropout=dropout)
  # Forward method
  def forward(self,x):
    embedding = self.embedding(x)
    output,hidden = self.rnn(embedding)

    return hidden


class Decoder(nn.Module):
  ''' This is the model class used to define the Decoder structure and to train the neural network on given parameters'''
  def __init__(self,output_size,embedding_size,hidden_size,layers,cell_type,bidirectional,dropout):
    '''
      output_size: The size of the output vocabulary or the number of distinct tokens.
      embedding_size: The size of the embedding vectors for each token.
      hidden_size: The size of the hidden state of the RNN cell.
      layers: The number of layers in the RNN.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      bidirectional: A boolean flag indicating whether the RNN is bidirectional or not.
      dropout: The dropout probability for the RNN layers
    '''
    super(Decoder,self).__init__()
    self.cell_type = cell_type
    self.embedding = nn.Embedding(output_size,embedding_size) 
    self.rnn = cell_type(embedding_size,hidden_size,layers,bidirectional = bidirectional,dropout=dropout)
    if bidirectional:
      self.out = nn.Linear(hidden_size*2,output_size) 
    else :
      self.out = nn.Linear(hidden_size,output_size) 

  # Forward method
  def forward(self,x,hidden):
    x = x.unsqueeze(1).transpose(0,1)
    embedding = self.embedding(x)
    output,hidden = self.rnn(embedding,hidden)
    output = self.out(output.squeeze(0))

    return output,hidden



# This is the model class used to define the seq2seq model with encoder and decoder structure and to train the neural network on given parameters
# Implements a sequence-to-sequence model using an encoder-decoder architecture and provides methods for training, validation, testing, and prediction. 
# It supports teacher forcing during training and beam search during inference 

class seq2seq(pl.LightningModule):
  def __init__(self,input_size,output_size,embedding_size,hidden_size,encoder_layer_size,decoder_layer_size,cell_type,beam_width,dropout,bidirectional,learning_rate=0.0001):
    '''
      input_size: The size of the input vocabulary or the number of distinct tokens.
      output_size: The size of the output vocabulary or the number of distinct tokens.
      embedding_size: The size of the embedding vectors for each token.
      hidden_size: The size of the hidden state of the RNN cells in the encoder and decoder.
      encoder_layer_size: The number of layers in the encoder.
      decoder_layer_size: The number of layers in the decoder.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      beam_width: The width of the beam search during inference.
      dropout: The dropout probability for the RNN layers.
      bidirectional: A boolean flag indicating whether the RNN layers are bidirectional or not.
      learning_rate: The learning rate for the optimizer (default is 0.0001)
    '''
    super(seq2seq,self).__init__()
    self.output_size = output_size
    self.lang_max_length = lang_max_length+2
    self.latin_max_length = latin_max_length+2
    self.cell_type = cell_type
    self.train_step_acc = []
    self.train_step_loss = []
    self.val_step_acc = []
    self.val_step_loss = []
    self.decoder_layer_size = decoder_layer_size #*  2 if bidirectional else 1
    self.bidirectional = bidirectional
    self.encoder_layer_size = encoder_layer_size 
    self.beam_width = beam_width
    self.encoder = Encoder(input_size,embedding_size,hidden_size,encoder_layer_size,cell_type,bidirectional,dropout)
    self.decoder = Decoder(output_size,embedding_size,hidden_size,decoder_layer_size,cell_type,bidirectional,dropout)
    self.learning_rate = learning_rate

  def beam_search(self,hidden,input,beam_width,output_len,output_seq):
    queue = []
    queue.append((input,1,hidden))
    for t in range(output_len):
      queue_temp = []
      for i in range(len(queue)):
        (input,prob_parent,hidden) = queue[i]
        output_rnn,hidden = self.decoder( input ,hidden)
        prob , index = torch.topk(output_rnn,beam_width)
        output_rnn = output_rnn.squeeze(1)
        for j in range(beam_width):
          prob_score = prob_parent*prob[0][j]
          if (prob_score > 1e-5):
            queue_temp.append((torch.tensor([index[0][j]]).to(device),prob_score,hidden))
      output_seq[t] = output_rnn
      queue = sorted(queue_temp,key = lambda x:x[1] , reverse = True)[:beam_width]
    return  output_seq

  def forward(self,input,output = [],tf = 0.5) :
    '''
    The forward method defines the forward pass of the seq2seq model. 
    It takes an input tensor input, an optional output tensor output, and a teacher-forcing ratio tf as inputs
    '''
    batch_size = input.shape[0]
    output_size = self.output_size

    bidir = 2 if self.bidirectional else 1
    hidden = self.encoder(input.transpose(0,1))
    if self.encoder_layer_size > self.decoder_layer_size :
      if (self.cell_type == nn.LSTM):
        (hidden,cell) = hidden
        hidden = hidden[-1*self.decoder_layer_size* bidir :]
        cell = cell[-1*self.decoder_layer_size*  bidir:]
        hidden = (hidden,cell) 
      else:
        hidden = hidden[-1*self.decoder_layer_size* bidir:]
        
    elif self.encoder_layer_size < self.decoder_layer_size :
      cell = []
      if (self.cell_type == nn.LSTM):
        (hidden,cell) = hidden
        temp_hidden = hidden[-1 *bidir:]
        temp_cell = cell[-1 *bidir:]
        if(temp_hidden.shape[0] == 0) :
            temp_hidden = hidden
            temp_cell = cell
        for i in range(self.decoder_layer_size - self.encoder_layer_size):   
          hidden = torch.cat((hidden,temp_hidden) )
          cell = torch.cat((cell,temp_cell))
        hidden = (hidden,cell) 
      else :
        temp_hidden = hidden[-1 *bidir:]
        if(temp_hidden.shape[0] == 0) :
            temp_hidden = hidden
        for i in range(self.decoder_layer_size - self.encoder_layer_size):   
          hidden = torch.cat((hidden,temp_hidden),dim=0 )
    
    output_seq = torch.zeros(self.lang_max_length,batch_size,output_size).to(device)
    if tf > 0:
      output = output.transpose(0,1)
    next_input = torch.tensor([2]*batch_size).to(device)
    


    # Iterates over the decoding steps,probabilities for each time step. The output probabilities are stored in the output sequence tensor
    if self.beam_width == 1:
      for t in range(self.lang_max_length):
        output_rnn,hidden = self.decoder( next_input ,hidden)
        output_seq[t] = output_rnn.squeeze(1)
        next_input =  output_seq[t].argmax(1) if  tf < torch.rand(1).item() else output[:,t]
      return output_seq
    else :
      for i in range(batch_size):
        output_seq_temp = torch.zeros(output.shape[1],1,output_size).to(device)
        # print(hidden.shape,next_input.shape,batch_size)
        # print(hidden[:,i,:]..shape)
        output_seq_temp = self.beam_search(hidden[:,i:i+1,:],next_input[i:i+1],self.beam_width,output.shape[1],output_seq_temp)
        output_seq[:,i:i+1,:] = output_seq_temp
      return output_seq


  def training_step(self,batch):
    '''
    The training_step method is called during the training loop for each batch of data. 
    It takes a batch of input-output pairs as input
    '''
    input,output = batch

    output = output.permute(1,0)
    # Calling forward 
    output_seq = self(input,output)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1
    output_dim = output_seq.shape[-1]
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)

    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)

    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('train_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.train_step_loss.append(loss)
    self.log('train_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    self.train_step_acc.append(acc)

    return loss



  def on_train_epoch_end(self):
    '''
    The on_train_epoch_end method is called at the end of each training epoch. 
    It calculates the average training loss and accuracy over the epoch, as well as the average validation loss and accuracy .
    '''
    train_acc =  torch.stack(self.train_step_acc).mean()
    train_loss =  torch.stack(self.train_step_loss).mean()
    val_acc =  torch.stack(self.val_step_acc).mean()
    val_loss =  torch.stack(self.val_step_loss).mean()
    print("train_loss:",train_loss.item(),"train_acc",train_acc.item(),"val_loss:",val_loss.item(),"val_acc",val_acc.item())
    wandb.log({"train_loss":train_loss.item(),"train_acc":train_acc.item(),"val_loss":val_loss.item(),"val_acc":val_acc.item()})
    self.train_step_acc.clear() 
    self.train_step_loss.clear() 
    self.val_step_acc.clear() 
    self.val_step_loss.clear() 


  def validation_step(self, batch,batch_idx):
    '''
    The validation_step method is similar to the training_step method but is called during the validation loop for each batch of data.
    It computes the loss and accuracy metrics for the validation data
    '''
    input,output = batch
    output = output.permute(1,0)
    output_seq = self(input,output,0)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1

    output_dim = output_seq.shape[-1]
  
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)
    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)
    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('val_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_loss.append(loss)
    self.log('val_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_acc.append(acc)

    return loss

  def configure_optimizers(self):
    '''
    The configure_optimizers method is used to configure the optimizer for the model. 
    In this case, it returns an Adam optimizer with the learning rate specified during initialization.
    '''
    return torch.optim.Adam(self.parameters(),lr= self.learning_rate)


  def test_step(self, batch,batch_idx):
    '''
    The test_step method is similar to the validation_step method but is called during the testing phase for each batch of data.
     It computes the loss and accuracy metrics for the test data
    '''
    input,output = batch
    output = output.permute(1,0)

    # Calling forward 
    output_seq = self(input,output,0)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1

    output_dim = output_seq.shape[-1]  
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)

    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)

    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('test_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.log('test_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    return loss

  def predict_step(self, batch,batch_idx,dataloader_idx=0):
    '''
    The predict_step method is called during the prediction phase for each batch of data.
    It takes the input batch and returns the predicted output sequence.
    '''
    trainX = batch
    output_seq = self(trainX)
    return output_seq


class AttnEncoder(nn.Module):
  ''' This is the model class used to define the Encoder structure and to train the neural network on given parameters'''
  def __init__(self,input_size,embedding_size,hidden_size,layers,cell_type,bidirectional,dropout):
    '''
      input_size: The size of the input vocabulary or the number of distinct tokens.
      embedding_size: The size of the embedding vectors for each token.
      hidden_size: The size of the hidden state of the RNN cell.
      layers: The number of layers in the RNN.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      bidirectional: A boolean flag indicating whether the RNN is bidirectional or not.
      dropout: The dropout probability for the RNN layers
    '''
    super(AttnEncoder,self).__init__()
    self.cell_type = cell_type
    self.embedding = nn.Embedding(input_size,embedding_size)
    self.rnn = cell_type(embedding_size,hidden_size,layers,bidirectional = bidirectional,dropout=dropout)
  # Forward method
  def forward(self,x):
    embedding = self.embedding(x)
    output,hidden = self.rnn(embedding)
    return output,hidden



class AttnDecoder(nn.Module):
  ''' This is the model class used to define the AttnDecoder structure and to train the neural network on given parameters'''
  def __init__(self,output_size,embedding_size,hidden_size,layers,cell_type,bidirectional,dropout,max_length):
    '''
      output_size: The size of the decoder output vocabulary.
      embedding_size: The size of the embedding layer.
      hidden_size: The size of the hidden state in the decoder.
      layers: The number of layers in the decoder.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      bidirectional: A boolean flag indicating whether the decoder is bidirectional or not.
      dropout: The dropout rate.
      max_length: The maximum length of input sequences.
    '''
    super(AttnDecoder,self).__init__()
    self.cell_type = cell_type

    self.embedding = nn.Embedding(output_size,embedding_size) 

    self.attn = nn.Linear(hidden_size+embedding_size , max_length)

    self.rnn = cell_type(embedding_size,hidden_size,layers,bidirectional = bidirectional,dropout=dropout)

    #  A linear layer that combines the attention context vector with the hidden state.
    if bidirectional:
      self.out = nn.Linear(hidden_size*2,output_size) 
      self.attn_combine = nn.Linear(hidden_size*2 +embedding_size, hidden_size)
    else :
      self.out = nn.Linear(hidden_size,output_size) 
      self.attn_combine = nn.Linear(hidden_size +embedding_size, hidden_size)

    
  def forward(self,x,hidden,encoder_output):
    x = x.unsqueeze(1).transpose(0,1)

    embedding = self.embedding(x)
    if(self.cell_type == nn.LSTM) :
      (hidden1,cell) = hidden
    else:
      hidden1 = hidden
    # Computes attention weights by concatenating the embedded input and the current hidden state and passing it through the attn linear layer
    attn_weights = functional.softmax(
      self.attn(torch.cat((embedding[0],hidden1[0]) , 1)) , dim =1)
    
    attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                           encoder_output.permute(1,0,2) )
    # Concatenates the embedded input and the attention context vector and passes it through the attn_combine linear layer.   
    output = torch.cat((embedding[0], attn_applied.squeeze(1)), 1)

    output = self.attn_combine(output).unsqueeze(0)

    output = functional.relu(output)

    output,hidden = self.rnn(embedding,hidden)
    
    output = self.out(output.squeeze(0))
    # attn_weights=attn_weights.permute(1,0)
    return output,hidden,attn_weights



class Attn2seq(pl.LightningModule):
  def __init__(self,input_size,output_size,embedding_size,hidden_size,encoder_layer_size,decoder_layer_size,cell_type,beam_width,dropout,bidirectional,learning_rate=0.0001):
    '''
      input_size: The size of the input vocabulary or the number of distinct tokens.
      output_size: The size of the output vocabulary or the number of distinct tokens.
      embedding_size: The size of the embedding vectors for each token.
      hidden_size: The size of the hidden state of the RNN cells in the encoder and decoder.
      encoder_layer_size: The number of layers in the encoder.
      decoder_layer_size: The number of layers in the decoder.
      cell_type: The type of the RNN cell (nn.RNN, nn.GRU, nn.LSTM).  
      beam_width: The width of the beam search during inference.
      dropout: The dropout probability for the RNN layers.
      bidirectional: A boolean flag indicating whether the RNN layers are bidirectional or not.
      learning_rate: The learning rate for the optimizer (default is 0.0001)
    '''
    super(Attn2seq,self).__init__()
    self.output_size = output_size
    self.input_size = input_size
    self.lang_max_length = lang_max_length+2
    self.latin_max_length = latin_max_length+2
    self.cell_type = cell_type
    self.train_step_acc = []
    self.train_step_loss = []
    self.val_step_acc = []
    self.val_step_loss = []
    self.decoder_layer_size = decoder_layer_size #*  2 if bidirectional else 1
    self.bidirectional = bidirectional
    self.encoder_layer_size = encoder_layer_size 
    self.beam_width = beam_width
    self.encoder = AttnEncoder(input_size,embedding_size,hidden_size,encoder_layer_size,cell_type,bidirectional,dropout)
    self.decoder = AttnDecoder(output_size,embedding_size,hidden_size,decoder_layer_size,cell_type,bidirectional,dropout,latin_max_length+2)
    self.learning_rate = learning_rate

  def beam_search(self,hidden,input,beam_width,output_len,output_seq):
    queue = []
    queue.append((input,1,hidden))
    for t in range(output_len):
      queue_temp = []
      for i in range(len(queue)):
        (input,prob_parent,hidden) = queue[i]
        output_rnn,hidden = self.decoder( input ,hidden)
        prob , index = torch.topk(output_rnn,beam_width)
        output_rnn = output_rnn.squeeze(1)
        for j in range(beam_width):
          prob_score = prob_parent*prob[0][j]
          if (prob_score > 1e-5):
            queue_temp.append((torch.tensor([index[0][j]]).to(device),prob_score,hidden))
      output_seq[t] = output_rnn
      queue = sorted(queue_temp,key = lambda x:x[1] , reverse = True)[:beam_width]
    return  output_seq

  def forward(self,input,output=[],tf = 0.5) :
    '''
    The forward method defines the forward pass of the Attn2seq model. 
    It takes an input tensor input, an optional output tensor output, and a teacher-forcing ratio tf as inputs
    '''
    batch_size = input.shape[0]
    output_size = self.output_size
    input_size = self.input_size

    bidir = 2 if self.bidirectional else 1
    
    output_encoder,hidden = self.encoder(input.transpose(0,1))

    output_seq = torch.zeros(self.lang_max_length,batch_size,output_size).to(device)
    attn_weights = torch.zeros(self.lang_max_length,batch_size,self.latin_max_length).to(device)

    if tf > 0:
      output = output.transpose(0,1)
    next_input = torch.tensor([2]*batch_size).to(device)
    for t in range(self.lang_max_length):
      output_rnn,hidden,attn_weights[t] = self.decoder( next_input ,hidden,output_encoder)
      output_seq[t] = output_rnn.squeeze(1)
      next_input =  output_seq[t].argmax(1) if  tf < torch.rand(1).item() else output[:,t]
    return output_seq,attn_weights

  def training_step(self,batch):
    '''
    The training_step method is called during the training loop for each batch of data. 
    It takes a batch of input-output pairs as input
    '''
    input,output = batch
    output = output.permute(1,0)

    # Calling forward 
    output_seq,attn_weights= self(input,output)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1
    output_dim = output_seq.shape[-1]
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)
    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)

    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('train_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.train_step_loss.append(loss)
    self.log('train_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    self.train_step_acc.append(acc)

    return loss



  def on_train_epoch_end(self):
    '''
    The on_train_epoch_end method is called at the end of each training epoch. 
    It calculates the average training loss and accuracy over the epoch, as well as the average validation loss and accuracy .
    '''
    train_acc =  torch.stack(self.train_step_acc).mean()
    train_loss =  torch.stack(self.train_step_loss).mean()
    val_acc =  torch.stack(self.val_step_acc).mean()
    val_loss =  torch.stack(self.val_step_loss).mean()
    print("train_loss:",train_loss.item(),"train_acc",train_acc.item(),"val_loss:",val_loss.item(),"val_acc",val_acc.item())
    wandb.log({"train_loss":train_loss.item(),"train_acc":train_acc.item(),"val_loss":val_loss.item(),"val_acc":val_acc.item()})
    self.train_step_acc.clear() 
    self.train_step_loss.clear() 
    self.val_step_acc.clear() 
    self.val_step_loss.clear() 


  def validation_step(self, batch,batch_idx):
    '''
    The validation_step method is similar to the training_step method but is called during the validation loop for each batch of data.
    It computes the loss and accuracy metrics for the validation data
    '''
    input,output = batch
    output = output.permute(1,0)
    # Calling forward 
    output_seq,attn_weights = self(input,output,0)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1

    output_dim = output_seq.shape[-1]
  
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)

    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)

    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('val_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_loss.append(loss)
    self.log('val_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_acc.append(acc)

    return loss

    

  def configure_optimizers(self):
    '''
    The configure_optimizers method is used to configure the optimizer for the model. 
    In this case, it returns an Adam optimizer with the learning rate specified during initialization.
    '''
    return torch.optim.Adam(self.parameters(),lr= self.learning_rate)



  def test_step(self, batch,batch_idx):
    '''
    The test_step method is similar to the training_step method but is called during the testing loop for each batch of data.
    It computes the loss and accuracy metrics for the testing data
    '''
    input,output = batch
    output = output.permute(1,0)
    # Calling forward 
    output_seq,attn_weights = self(input,output,0)
    output = output.permute(1,0)

    output_seq_2 = torch.zeros(output_seq.shape).to(device)
    batch_n = np.arange(len(output_seq))

    for f in range(len(output)):
      col = output[f]
      output_seq_2[batch_n,f,np.array(col.cpu())] = 1

    output_dim = output_seq.shape[-1]
  
    output_seq_t1 = output_seq[1:].view(-1,output_dim)
    output_seq_t2 = output_seq_2[1:].view(-1,output_dim)

    # Computing the cross-entropy loss between the predicted output sequence and the true output sequence
    loss = nn.CrossEntropyLoss()
    loss = loss(output_seq_t1,output_seq_t2).mean()
    output = output.permute(1,0)

    # Computing the accuracy by comparing the predicted output with the true output.
    output_ = torch.argmax(output_seq,2)
    acc_1 = torch.all(output_[1:-1,:] == output[1:-1,:],dim=0)
    acc = torch.sum(acc_1 == True)/len(acc_1)

    # Logging the loss and accuracy metrics using the self.log method.
    self.log('val_loss', loss,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_loss.append(loss)
    self.log('val_acc', acc,on_epoch = True,on_step = False,prog_bar=True)
    self.val_step_acc.append(acc)

    return loss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project',required = False,metavar="",default ='Assignment-031',type=str,help = "Project name used to track experiments in Weights & Biases dashboard" )
parser.add_argument('-we','--wandb_entity',required = False,metavar="",default ='saisreeram',type=str,help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument('-dp','--dataset_path',required = False,metavar="",default ='aksharantar_sampled/',type=str,help = 'Give folder name with /')
parser.add_argument('-la','--lang',required = False,metavar="",default ='hin',type=str,help = 'language name given in folder /')

parser.add_argument('-e','--epochs',required = False,metavar="",default =1,type=int,help = "Number of epochs to train the model." )
parser.add_argument('-b','--batch_size',required = False,metavar="",default =16,type=int,help = "Batch size used to train the model.")

parser.add_argument('-lr','--learning_rate',required = False,metavar="",default =0.0001,type=float,help = "Learning rate used to optimize model parameters" )
parser.add_argument('-cl','--cell_type',required = False,metavar="",default ='LSTM',type=str,choices = ["RNN", "GRU", "LSTM"],help = 'choices: ["RNN", "GRU", "LSTM"]')

parser.add_argument('-bd','--bidirectional',required = False,metavar="",default =True,type=str,choices = [True,False],help = 'choices: [True,False]')
parser.add_argument('-do','--dropout',required = False,metavar="",default =0.4,type=float,help = 'Value of dropout ')

parser.add_argument('-at','--isattention',required = False,metavar="",default =True,type=str,choices = [True,False],help = 'choices: [True,False]')

parser.add_argument('-el','--encoder_layers',required = False,metavar="",default =3,type=int,help = 'The number of layers in the encoder ')
parser.add_argument('-dl','--decoder_layers',required = False,metavar="",default =2,type=int,help = 'The number of layers in the decoder')

parser.add_argument('-es','--embedding_size',required = False,metavar="",default =64,type=int,help = "The size of the embedding vectors for each token" )
parser.add_argument('-hls','--hidden_layer_size',required = False,metavar="",default =128,type=int,help = "The size of the hidden state of the RNN cells in the encoder and decoder" )


args = parser.parse_args()
wandb_entity = args.wandb_entity
wandb_project = args.wandb_project
dataset_path = args.dataset_path
lang = args.lang
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
bidirectional = args.bidirectional
dropout = args.dropout
cell_type = args.cell_type
encoder_layers = args.encoder_layers
decoder_layers = args.decoder_layers
embedding_size = args.embedding_size
hidden_layer_size = args.hidden_layer_size
isattention = args.isattention
cell_map = {"RNN":nn.RNN, "GRU":nn.GRU, "LSTM":nn.LSTM}
cell_type = cell_map[cell_type]

wandb.login(key = "8d6c17aa48af2229c26cbc16513ef266358c0b96")
wandb.init(project=wandb_project,entity = wandb_entity)

# lang = "hin" # name of the lang to train the model
base_dir = dataset_path # root dir containing all the lang datasets

base_dir = base_dir+lang+"/" 
# hindi_font = FontProperties(fname = 'hindi/TiroDevanagariHindi-Regular.ttf')

train_file = base_dir+lang+"_train.csv"
val_file = base_dir+lang+"_valid.csv"
test_file = base_dir+lang+"_test.csv"

# loading train,valid,test csv files
train_data = pd.read_csv(train_file,header=None)
val_data = pd.read_csv(val_file,header=None)
test_data = pd.read_csv(test_file,header=None)


# Initialization of special tokens in Character dictionaries
latin_chars = {'<PAD>': 0, '<UNK>': 1,'<start>':2,'<end>':3}
lang_chars = {'<PAD>': 0, '<UNK>': 1,'<start>':2,'<end>':3}
latin_chars_num = {}
lang_chars_num = {}

# Character counting and numbering
for word in train_data[0]:
  for char in word :
    if char not in latin_chars:
      latin_chars[char] = len(latin_chars)

for word in train_data[1]:
  for char in word :
    if char not in lang_chars:
      lang_chars[char] = len(lang_chars)

# Reverse dictionaries
for key,value in latin_chars.items():
  latin_chars_num[value] = key

for key,value in lang_chars.items():
  lang_chars_num[value] = key

# Maximum word length
latin_max_length = len(max(train_data[0],key = len))
lang_max_length = len(max(train_data[1],key = len))

#Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def word_to_vec(data):
  data1= data.T
  data_pairs = []
  for i in range(0,len(data)):  # Iterating over data
    # Encoding words
    word =  [2]+[latin_chars.get(char,latin_chars['<UNK>']) for char in data1[i][0]] + [0]*(latin_max_length - len(data1[i][0]))+[3]
    latin_tensor = torch.tensor(word).to(device)
    word =  [2]+[lang_chars.get(char,lang_chars['<UNK>']) for char in data1[i][1]] + [0]*(lang_max_length - len(data1[i][1]))+[3]
    lang_tensor = torch.tensor(word).to(device)
    # Building data pairs
    data_pairs.append([latin_tensor,lang_tensor])
  # Returning data pairs
  return data_pairs 

# Conversion to Encoded word pairs
train_data_pairs = word_to_vec(train_data)
val_data_pairs = word_to_vec(val_data)
test_data_pairs = word_to_vec(test_data)

# Data loaders and Batch size and shuffling
train_dataloader = DataLoader(train_data_pairs, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data_pairs, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data_pairs, batch_size=batch_size, shuffle=False)


if isattention == True :
    model = Attn2seq(input_size = len(latin_chars), output_size = len(lang_chars),
                embedding_size = embedding_size, hidden_size = hidden_layer_size,encoder_layer_size = encoder_layers,
                decoder_layer_size = encoder_layers,cell_type = cell_type,
                beam_width = 1,dropout= dropout,
                bidirectional =bidirectional ,learning_rate = learning_rate)

else :
    model = seq2seq(input_size = len(latin_chars), output_size = len(lang_chars),
                embedding_size = embedding_size, hidden_size = hidden_layer_size,encoder_layer_size = encoder_layers,
                decoder_layer_size = decoder_layers,cell_type = cell_type,
                beam_width = 1,dropout= dropout,
                bidirectional =bidirectional ,learning_rate = learning_rate)
model.to(device)

trainer = pl.Trainer(max_epochs=epochs,accelerator = 'gpu') 
trainer.fit(model, train_dataloader,val_dataloader)

wandb.finish()
