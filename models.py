import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import sys
import numpy
import math


# BiRNN Model (Many-to-One)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True) 
        self.fc = nn.Linear(self.hidden_size*2, num_classes)  # 2 for bidirection 
    
    def forward(self, sentence, train = True):
        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.dropout(embeds)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
        
        # Forward propagate RNN
        out, _ = self.lstm(embeds, (h0, c0))

        # Decode hidden state of last time step
        if train:
            out = self.fc(self.dropout(out[:, -1, :]))
        else:
            out = self.fc(out[:, -1, :])

        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, dropout_rate, Ci = 1, kernel_num = 100, \
        kernel_sizes=[3,4,5]):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.padding = nn.ReflectionPad2d((0,0,1,1))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embedding_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.highway_t = nn.Linear(len(kernel_sizes)*kernel_num, len(kernel_sizes)*kernel_num) # square matrix
        self.highway_g = nn.Linear(len(kernel_sizes)*kernel_num, len(kernel_sizes)*kernel_num) # square matrix
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)

    def highway(self, input_, num_layers=1, bias=-2.0):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        for idx in range(num_layers):
            t = F.sigmoid(self.highway_t(input_) + bias)
            g = F.relu(self.highway_g(input_))

            output = t * g + (1. - t) * input_
            input_ = output

        return output


    def forward(self, x, train = True):
        x = self.embed(x) # (N,W,D)

        x = x.unsqueeze(1) # (N,Ci,W,D) # N 은 뱃치수, Ci 가 채널수, W 가 단어수(윈도우수)- 3개보다 작으면 에러남, D가 embedding_size 
        x = self.padding(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.highway(x, 1, 0)

        if train:
            x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit


class Cha_CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, dropout_rate, Ci = 1, kernel_num = 100, \
        kernel_sizes=[3,4,5]):
        super(Cha_CNN_LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.padding = nn.ReflectionPad2d((0,0,1,1))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embedding_size)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, num_classes)

    def forward(self, x, train = True):
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
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(Siamese_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True) 

        self.LogisticRegression = nn.Linear(self.hidden_size*2*2, num_classes) # 2 for bidirection, 2 for siamese

    
    def siamese(self, sentence, train = True):
        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.dropout(embeds)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
        
        # Forward propagate BRNN
        out, _ = self.lstm(embeds, (h0, c0))

        # Decode hidden state of last time step
        if train:
            out = self.dropout(out[:, -1, :])
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
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, num_classes)  # *2 for bidirection, BiLSTM 을 오토인코더로 초기화할 때 필요.

    def initHidden(self, batch_size):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size)).cuda()
        return (h0, c0)

    
    def forward(self, sentence, train = True):
        #print(sentence)
        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.dropout(embeds)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda() # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        output, hidden = self.lstm(embeds, (h0, c0))

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.to_decoder_init_hidden = nn.Linear(2*num_layers, num_layers)

        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(self.hidden_size, num_classes)  # 2 for bidirection 


    def initHidden(self, batch_size, init_hidden):
        # Set initial states
        decoder_init_hidden = self.to_decoder_init_hidden(init_hidden[0].transpose(2, 0)).transpose(2, 0).contiguous()
        decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        return (decoder_init_hidden, decoder_init_c)

    
    def forward(self, sentence, hidden, train = True):
        embeds = self.word_embedding(sentence)
        if train:
            embeds = self.dropout(embeds)

        # Forward propagate RNN
        #embeds = F.relu(embeds)
        output, hidden = self.lstm(embeds, hidden)
        
        # Decode hidden state of last time step
        if train:
            output = self.fc(self.dropout(output[:, -1, :]))
        else:
            output = self.fc(output[:, -1, :])

        return output, hidden


# from pytorch tutorial http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate, max_sequence_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_sequence_length = max_sequence_length
        
        self.W_A = nn.Linear(embedding_size + 2*self.hidden_size, self.max_sequence_length)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.W_C = nn.Linear(embedding_size + 2*self.hidden_size, 2*self.hidden_size)
        self.lstm = nn.LSTM(2*self.hidden_size, 2*self.hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(2*self.hidden_size, num_classes)


    def initHidden(self, batch_size, init_hidden):
        # Set initial states
        tmp = torch.unbind(init_hidden[0], dim=0)
        decoder_init_hidden = torch.cat(tmp, dim=1).unsqueeze(0)
        decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, 2*self.hidden_size)).cuda()
        return (decoder_init_hidden, decoder_init_c)


    def forward(self, input, hidden, encoder_outputs, train = True):
        embeds = self.word_embedding(input)
        if train:
            embeds = self.dropout(embeds)

        encoder_outputs_max = Variable(torch.zeros( embeds.size(0), self.max_sequence_length, 2*self.hidden_size)).cuda()
        encoder_outputs_max[:, :encoder_outputs.size(1), :] = encoder_outputs
        
        score = self.W_A(torch.cat((embeds, hidden[0].transpose(0,1)), 2))
        attn_weights = F.softmax(score, dim=2)
        context = torch.bmm(attn_weights, encoder_outputs_max)
        
        output = torch.cat((embeds, context), 2)
        output = F.relu(self.W_C(output)) # eq(5)
        
        output, hidden = self.lstm(output, hidden)
        
        # Decode hidden state of last time step
        if train:
            output = self.fc(self.dropout(output[:, -1, :])) # eq (6)
        else:
            output = self.fc(output[:, -1, :])

        return output, hidden, attn_weights


# # from Effective Approaches to Attention-based Neural Machine Translation https://arxiv.org/pdf/1508.04025.pdf.
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate, max_sequence_length):
#         super(AttnDecoderRNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         self.word_embedding = nn.Embedding(vocab_size, embedding_size)
#         self.W_C = nn.Linear(embedding_size + 2*self.hidden_size, 2*self.hidden_size)
#         self.lstm = nn.LSTM(2*self.hidden_size, 2*self.hidden_size, num_layers, batch_first=True) 
#         self.fc = nn.Linear(2*self.hidden_size, num_classes)


#     def initHidden(self, batch_size, init_hidden):
#         # Set initial states
#         tmp = torch.unbind(init_hidden[0], dim=0)
#         decoder_init_hidden = torch.cat(tmp, dim=1).unsqueeze(0)
#         decoder_init_c = Variable(torch.zeros(self.num_layers, batch_size, 2*self.hidden_size)).cuda()
#         return (decoder_init_hidden, decoder_init_c)


#     def forward(self, input, hidden, encoder_outputs, train = True):
#         embeds = self.word_embedding(input)
#         if train:
#             embeds = self.dropout(embeds)
        
#         W_enc= encoder_outputs # for simple dot product attention
#         score = torch.bmm(hidden[0].transpose(0,1), W_enc.transpose(1,2)) # eq(7), dot product
#         attn_weights = F.softmax(score, dim=2)
#         context = torch.bmm(attn_weights, encoder_outputs)
        
#         output = torch.cat((embeds, context), 2)
#         output = F.tanh(self.W_C(output)) # eq(5)
#         #output = F.relu(self.W_C(output)) # eq(5)
#         
#         output, hidden = self.lstm(output, hidden)
        
#         # Decode hidden state of last time step
#         if train:
#             output = self.fc(self.dropout(output[:, -1, :])) # eq (6)
#         else:
#             output = self.fc(output[:, -1, :])
#         return output, hidden, attn_weights

