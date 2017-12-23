import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import sys
import numpy
import math


# BiRNN Model (Many-to-One)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, word_dropout_rate, dropout_rate):
        super(BiLSTM, self).__init__()
        self.word_dropout_rate = word_dropout_rate
        self.hidden_size = hidden_size
        self.word_dropout = nn.Dropout(p=word_dropout_rate)
        self.vertical_dropout = nn.Dropout(p=dropout_rate)

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True) 
        self.fc = nn.Linear(self.hidden_size*2, num_classes)  # 2 for bidirection 
    
    def forward(self, sentence, train = True):
        if train:
            sentence = self.word_dropout(sentence) * (1 - self.word_dropout_rate)
        sentence = sentence.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.

        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.vertical_dropout(embeds)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
        
        # Forward propagate RNN
        out, _ = self.lstm(embeds, (h0, c0))

        # Decode hidden state of last time step
        if train:
            out = self.fc(self.vertical_dropout(out[:, -1, :]))
        else:
            out = self.fc(out[:, -1, :])

        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, word_dropout_rate, dropout_rate, Ci = 1, kernel_num = 100, \
        kernel_sizes=[3,4,5]):
        super(CNN, self).__init__()
        self.word_dropout_rate = word_dropout_rate
        self.word_dropout = nn.Dropout(word_dropout_rate) 
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.padding = nn.ReflectionPad2d((0,0,1,1))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embedding_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)

    def forward(self, x, train = True):
        if train:
            x = self.word_dropout(x.float()) * (1 - self.word_dropout_rate)
        x = x.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.
        x = self.embed(x) # (N,W,D)

        x = x.unsqueeze(1) # (N,Ci,W,D) # N 은 뱃치수, Ci 가 채널수, W 가 단어수(윈도우수)- 3개보다 작으면 에러남, D가 embedding_size 
        x = self.padding(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        if train:
            x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit


class Cha_CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, word_dropout_rate, dropout_rate, Ci = 1, kernel_num = 100, \
        kernel_sizes=[3,4,5]):
        super(Cha_CNN_LSTM, self).__init__()
        self.word_dropout_rate = word_dropout_rate
        self.word_dropout = nn.Dropout(word_dropout_rate) 
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.padding = nn.ReflectionPad2d((0,0,1,1))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embedding_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)

    def forward(self, x, train = True):
        if train:
            x = self.word_dropout(x) * (1 - self.word_dropout_rate)
        x = x.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.
        x = self.embed(x) # (N,W,D)

        x = x.unsqueeze(1) # (N,Ci,W,D) # N 은 뱃치수, Ci 가 채널수, W 가 단어수(윈도우수)- 3개보다 작으면 에러남, D가 embedding_size 
        x = self.padding(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        if train:
            x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit


class Siamese_CNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_size=100, max_sequence_length = 30, kernel_size = 3, kernel_num=32):
        super(Siamese_CNN, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.kernel_size = kernel_size

        #ABCNN 논문에서 BCNN 의 네트웍 구조를 빌림.
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d((0,0,self.kernel_size-1,self.kernel_size-1)),
            nn.Conv2d(embedding_size, kernel_num, (self.kernel_size, 1)),
            nn.ReLU(),
            nn.AvgPool2d((self.kernel_size,1), stride=1))
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d((0,0,self.kernel_size-1,self.kernel_size-1)),
            nn.Conv2d(kernel_num, kernel_num, (self.kernel_size, 1)),
            nn.ReLU(),
            nn.AvgPool2d((max_sequence_length+2,1), stride=1))
        
        self.LogisticRegression = nn.Linear(kernel_num*2, num_classes)

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.sentence_pad = nn.ReflectionPad2d((0,0,0,1))

        
    def siamese(self, x):
        x = self.embed(x) # (N,W,D)
        x = x.unsqueeze(1) # (N,Ci,W,D) # N 은 뱃치수, Ci 가 채널수, W 가 단어수(윈도우수)- 3개보다 작으면 에러남, D가 embedding_size 
        x = x.permute(0,3,2,1) # (N,D,W,Ci) # 이렇게 바꿔두자.
        if x.size(2)>self.max_sequence_length:
            x=x[:,:,:self.max_sequence_length,:]
        while (x.size(2)<self.max_sequence_length):
             x = self.sentence_pad(x)
    
        #print(x.size())
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = out.view(out.size(0), -1)
        
        return out

    def forward(self, sentence_1, sentence_2, train = True):
        represent_1 = self.siamese(sentence_1)
        represent_2 = self.siamese(sentence_2)
        similarity = torch.cat([represent_1, represent_2],1)
        logit = self.LogisticRegression(similarity) 

        return logit


# Siamese_BiLSTM Model (Many-to-One)
class Siamese_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, word_dropout_rate, dropout_rate):
        super(Siamese_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.word_dropout_rate = word_dropout_rate
        self.word_dropout = nn.Dropout(p=word_dropout_rate)
        self.vertical_dropout = nn.Dropout(p=dropout_rate)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True) 

        self.LogisticRegression = nn.Linear(self.hidden_size*2*2, num_classes) # 2 for bidirection, 2 for siamese

    
    def siamese(self, sentence, train = True):
        if train:
            sentence = self.word_dropout(sentence) * (1 - self.word_dropout_rate)
        sentence = sentence.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.

        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.vertical_dropout(embeds)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
        
        # Forward propagate BRNN
        out, _ = self.lstm(embeds, (h0, c0))

        # Decode hidden state of last time step
        if train:
            out = self.vertical_dropout(out[:, -1, :])
        else:
            out = out[:, -1, :]

        return out

    def forward(self, sentence_1, sentence_2, train = True):
        represent_1 = self.siamese(sentence_1, train)
        represent_2 = self.siamese(sentence_2, train)
        similarity = torch.cat([represent_1, represent_2], 1)
        logit = self.LogisticRegression(similarity) 

        return logit



class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, word_dropout_rate, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.word_dropout_rate = word_dropout_rate
        self.hidden_size = hidden_size
        self.word_dropout = nn.Dropout(p=word_dropout_rate)
        self.vertical_dropout = nn.Dropout(p=dropout_rate)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True) 
    
    def forward(self, sentence, init_hidden, train = True):
        if train:
            sentence = self.word_dropout(sentence) * (1 - self.word_dropout_rate)
        sentence = sentence.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.
        
        embeds = self.word_embedding(sentence)
        
        if train:
            embeds = self.vertical_dropout(embeds)

        # Forward propagate RNN
        output, hidden = self.lstm(embeds, init_hidden)

        return output, hidden

    def initHidden(self, batch_size):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size)).cuda()
        return (h0, c0)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, word_dropout_rate, dropout_rate):
        super(DecoderRNN, self).__init__()
        self.word_dropout_rate = word_dropout_rate
        self.hidden_size = hidden_size
        self.word_dropout = nn.Dropout(p=word_dropout_rate)
        self.vertical_dropout = nn.Dropout(p=dropout_rate)

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.to_decoder_init_hidden = nn.Linear(2*num_layers, num_layers)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(self.hidden_size, num_classes)  # 2 for bidirection 
    
    def forward(self, sentence, hidden, train = True):
        if train:
            sentence = self.word_dropout(sentence) * (1 - self.word_dropout_rate)
        sentence = sentence.long() # nn.Embedding 하기 위해 다시 int 형태로 바꿈.
        
        embeds = self.word_embedding(sentence)
        
        if train:
            embeds = self.vertical_dropout(embeds)

        # Forward propagate RNN
        #embeds = F.relu(embeds)
        output, hidden = self.lstm(embeds, hidden)

        # Decode hidden state of last time step
        if train:
            output = self.fc(self.vertical_dropout(output[:, -1, :]))
        else:
            output = self.fc(output[:, -1, :])

        return output, hidden


    def initHidden(self, batch_size, init_hidden):
        # Set initial states
        decoder_init_hidden = self.to_decoder_init_hidden(init_hidden[0].transpose(2, 0)).transpose(2, 0).contiguous()
        decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        return (decoder_init_hidden, decoder_init_c)



class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(AttnDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vertical_dropout = nn.Dropout(p=dropout_rate)
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        #self.to_decoder_init_hidden = nn.Linear(2*num_layers, num_layers) 
        
        self.lstm = nn.LSTM(embedding_size+ 2*self.hidden_size, 2*self.hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(2*self.hidden_size+2*self.hidden_size, num_classes)

    def forward(self, input, hidden, encoder_outputs, train = True):
        embeds = self.word_embedding(input)
        if train:
            embeds = self.vertical_dropout(embeds)
        
        #W_enc= self.general_attn_W(encoder_outputs)
        W_enc= encoder_outputs # for simple dot product attention

        attn_weights = F.softmax(torch.bmm(hidden[0].transpose(0,1), W_enc.transpose(1,2)).view(-1, W_enc.size(1))).view(embeds.size(0), -1, W_enc.size(1))
        context = torch.bmm(attn_weights, encoder_outputs)
        
        output = torch.cat((embeds, context), 2)
        
        #output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        
        output = torch.cat((output, context), 2)
        # Decode hidden state of last time step
        if train:
            output = self.fc(self.vertical_dropout(output[:, -1, :]))
        else:
            output = self.fc(output[:, -1, :])
        return output, hidden, attn_weights


    def initHidden(self, batch_size, init_hidden):
        # Set initial states
        #decoder_init_hidden = self.to_decoder_init_hidden(init_hidden[0].transpose(2, 0)).transpose(2, 0).contiguous()
        tmp = torch.unbind(init_hidden[0], dim=0)
        decoder_init_hidden = torch.cat(tmp, dim=1).unsqueeze(0)
        decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, 2*self.hidden_size)).cuda()
        return (decoder_init_hidden, decoder_init_c)



# general attention
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
#         super(AttnDecoderRNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.vertical_dropout = nn.Dropout(p=dropout_rate)
        
#         self.word_embedding = nn.Embedding(vocab_size, embedding_size)
#         self.to_decoder_init_hidden = nn.Linear(2*num_layers, num_layers)
        
#         self.general_attn_W = nn.Linear(2*hidden_size, hidden_size)
        
#         self.lstm = nn.LSTM(embedding_size+ 2*hidden_size, hidden_size, num_layers, batch_first=True) 
#         self.fc = nn.Linear(hidden_size+2*hidden_size, num_classes)

#     def forward(self, input, hidden, encoder_outputs, train = True):
#         embeds = self.word_embedding(input)
#         if train:
#             embeds = self.vertical_dropout(embeds)
        
#         W_enc= self.general_attn_W(encoder_outputs)
        
#         attn_weights = F.softmax(torch.bmm(hidden[0].transpose(0,1), W_enc.transpose(1,2)).view(-1, W_enc.size(1))).view(embeds.size(0), -1, W_enc.size(1))
#         context = torch.bmm(attn_weights, encoder_outputs)
        
#         output = torch.cat((embeds, context), 2)
        
#         #output = F.relu(output)
#         output, hidden = self.lstm(output, hidden)
        
#         output = torch.cat((output, context), 2)
#         # Decode hidden state of last time step
#         if train:
#             output = self.fc(self.vertical_dropout(output[:, -1, :]))
#         else:
#             output = self.fc(output[:, -1, :])
#         return output, hidden, attn_weights


#     def initHidden(self, batch_size, init_hidden):
#         # Set initial states
#         decoder_init_hidden = self.to_decoder_init_hidden(init_hidden[0].transpose(2, 0)).transpose(2, 0).contiguous()
#         decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
#         return (decoder_init_hidden, decoder_init_c)
