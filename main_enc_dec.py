import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sys
import numpy
import random

from utils import *
from models import *

# Hyper Parameters
max_sequence_length = 50
max_vocabulary_size = 25000
embedding_size = int(sys.argv[5]) #300
hidden_size = int(sys.argv[6])#256
num_layers = 1
#num_classes = 2
batch_size = 50
num_epochs = 100
learning_rate = 0.001 #float(sys.argv[6]) #0.003
dropout_rate = float(sys.argv[7]) #0

train_data_path = sys.argv[1] 
test_data_path = sys.argv[2] 
directory_name = sys.argv[3] #'171218gen'

encoder_name = 'ENC'
decoder_name = sys.argv[4] #'DEC'
model_name = encoder_name + '_' + decoder_name

word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=False)
#word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=True)
num_classes = vocab_size

encoder = EncoderRNN(vocab_size, embedding_size, hidden_size, num_layers, dropout_rate)
if decoder_name == 'AttDEC':
    decoder = AttnDecoderRNN(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate)
else:
    decoder = DecoderRNN(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate)

encoder = encoder.cuda()
decoder = decoder.cuda()
print(encoder)
print(decoder)

train_dataset = GenerateDataset(train_data_path, word_to_ix, batch_size)
valid_dataset = GenerateDataset(valid_data_path, word_to_ix, 1)
test_dataset = GenerateDataset(test_data_path, word_to_ix, 1)

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset)
test_loader = DataLoader(dataset=test_dataset)
print('length of train dataset', len(train_dataset)*batch_size)
print('length of valid dataset', len(valid_dataset))
print('length of test dataset', len(test_dataset))


teacher_forcing_ratio = 1
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_data = True):
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
       
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    SOS_token = 0
    SOS_token_batch = torch.zeros(batch_size, 1).long()
    decoder_input = Variable(torch.LongTensor(SOS_token_batch))
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder.initHidden(batch_size, encoder_hidden)

    decoder_input_list = torch.unbind(target_variable, dim=1)
    
    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        if decoder_name == 'AttDEC':
            for next_input in decoder_input_list:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, next_input)
                decoder_input = next_input.unsqueeze(1)
        else:
            for next_input in decoder_input_list:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, next_input)
                decoder_input = next_input.unsqueeze(1)
    else:
        if decoder_name == 'AttDEC':
            for next_input in decoder_input_list:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, next_input)
                
                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.cuda() # Chosen word is next input
                
        else:
            for next_input in decoder_input_list:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, next_input)
                
                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.cuda() # Chosen word is next input

    if train_data:
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 1)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 1)
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.data[0] / target_variable.size()[0]


def evaluate(encoder, decoder, input_variable):
    encoder_hidden = encoder.initHidden(1)

    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden, train = False)

    SOS_token = 0
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = encoder_hidden
    decoder_hidden = decoder.initHidden(1, encoder_hidden)

    decoded_words = []
    decoder_attentions = []

    if decoder_name == 'AttDEC':
        for _ in range(max_sequence_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, train = False)
            decoder_attentions.append(decoder_attention.data)

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoded_words.append(ni)
            if ni ==1: break

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([[ni]])).cuda()
    else:
        for _ in range(max_sequence_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, train = False)

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoded_words.append(ni)
            if ni ==1: break

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([[ni]])).cuda()
        
    if decoder_name == 'AttDEC':
        return decoded_words, decoder_attentions
    else:
        return decoded_words


print_every = (len(train_dataset)/2)

# Loss and Optimizer
loss_function = nn.CrossEntropyLoss() # = log softmax + NLL loss
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

print_loss_total = 0  
keep_valid_loss = 1e06
if not os.path.exists('./models'):
    os.makedirs('./models')
if not os.path.exists('./models/'+directory_name):
    os.makedirs('./models/'+directory_name)

for epoch in range(num_epochs):
    for i, (sentence_1, sentence_2) in enumerate(train_loader):
        start = time.time()
        sentence_1 = Variable(sentence_1.view(batch_size, -1)).cuda()
        sentence_2 = Variable(sentence_2.view(batch_size, -1)).cuda()
        loss = train(sentence_1, sentence_2, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function)
        print_loss_total += loss
        
        if ((i+1)+epoch*len(train_dataset)) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
            # valid set test
            valid_loss = 0
            correct = 0
            total = 0 
            for sentence_1, sentence_2 in valid_loader:
                loss = train(sentence_1, sentence_2, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, train_data = False)
                valid_loss += loss
                
            if valid_loss < keep_valid_loss:
                keep_valid_loss = valid_loss
                torch.save(encoder.state_dict(), './models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl' % (encoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio))
                torch.save(decoder.state_dict(), './models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl' % (decoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio))
                
                print('%s (%d %d%%) loss %.4f, valid loss: %.4f <saved model>' % (timeSince(start, ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) ),
                                         (i+1)+epoch*len(train_dataset), ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) * 100, print_loss_avg, valid_loss))
                
            else: 
                print('%s (%d %d%%) loss %.4f, valid loss: %.4f' % (timeSince(start, ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) ),
                                         (i+1)+epoch*len(train_dataset), ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) * 100, print_loss_avg, valid_loss))
            losses = 0


# Test the Model
correct = 0
total = 0
# load the best valid model.
encoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl' % (encoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio)))
decoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl' % (decoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio)))

if not os.path.exists('./result'):
    os.makedirs('./result')
if not os.path.exists('./result/'+directory_name):
    os.makedirs('./result/'+directory_name)
with open('./result/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_generation.txt'%(decoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio), 'a', encoding ='utf-8') as w:

    attention_result = []
    for sentence_1, sentence_2 in test_loader:
        sentence_variable = Variable(sentence_1.view(1, -1)).cuda()

        if decoder_name == 'AttDEC':
            predicted, attention = evaluate(encoder, decoder, sentence_variable)
            attention_result.append(attention)
        else:
            predicted = evaluate(encoder, decoder, sentence_variable)
        
        sentence_1 = [ix_to_word[word_idx] for word_idx in sentence_1.long()[0][0].numpy()]
        sentence_2 = [ix_to_word[word_idx] for word_idx in sentence_2.long()[0][0].numpy()]
        predicted = [ix_to_word[word_idx] for word_idx in predicted]
        w.write('Input_sentence: \t' + ' '.join(sentence_1)+'\n')
        w.write('Target_sentence: \t' + ' '.join(sentence_2)+'\n')
        w.write('Generated_sentence: \t' + ' '.join(predicted)+'\n')
    
    w.write('[setting] : '+'\temb_size\t'+str(embedding_size)+'\tHid\t'+str(hidden_size)+'\tD\t'+str(dropout_rate)+'\n')
    w.write('saved to ./models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl\n' % (encoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('saved to ./models/'+directory_name+'/%s_emb%d_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl\n' % (decoder_name, embedding_size, hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('-'*100+'\n\n')
