import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import sys
import numpy
import random

from utils import *
from models import *

# Hyper Parameters
max_sequence_length = 50
max_vocabulary_size = 25000
embedding_size = 300
hidden_size = int(sys.argv[6])#256
num_layers = 1
batch_size = 150
num_epochs = 50
learning_rate = 0.001
dropout_rate = float(sys.argv[7]) #0
teacher_forcing_ratio = float(sys.argv[8]) #0.5

train_data_path = sys.argv[1]
valid_data_path = sys.argv[2] 
test_data_path = sys.argv[3] # test_twitter.txt
directory_name = sys.argv[4] #'twit_gen'

encoder_name = 'AttENC'
decoder_name = sys.argv[5] #'DEC'

word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=False)
num_classes = vocab_size

encoder = EncoderRNN(vocab_size, embedding_size, hidden_size, num_layers, 2, dropout_rate)
if decoder_name == 'AttDEC':
    decoder = AttnDecoderRNN(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate, max_sequence_length)
else:
    decoder = DecoderRNN(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate)

encoder = encoder.cuda()
decoder = decoder.cuda()
print(encoder)
print(decoder)

if decoder_name == 'AE': #for autoencoder
    train_dataset = GenerateDataset_AE(train_data_path, word_to_ix, batch_size)
    valid_dataset = GenerateDataset_AE(valid_data_path, word_to_ix, 1)
    test_dataset = GenerateDataset_AE(test_data_path, word_to_ix, 1)
else:
    train_dataset = GenerateDataset(train_data_path, word_to_ix, batch_size)
    valid_dataset = GenerateDataset(valid_data_path, word_to_ix, 1)
    test_dataset = GenerateDataset(test_data_path, word_to_ix, 1)

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset)
test_loader = DataLoader(dataset=test_dataset)
print('length of train dataset', len(train_dataset)*batch_size)
print('length of valid dataset', len(valid_dataset))
print('length of test dataset', len(test_dataset))


def train(input_variable, target_variable, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_data = True):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
       
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable)

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
                decoder_input = Variable(topi).cuda() # Chosen word is next input
                
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
    encoder_outputs, encoder_hidden = encoder(input_variable, train=False)
    #max_sequence_length = encoder_outputs.size(1) # for attention-simple dot product mode
    
    SOS_token = 0
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = encoder_hidden
    decoder_hidden = decoder.initHidden(1, encoder_hidden)

    decoded_words = []
    decoder_attentions = torch.zeros(max_sequence_length, max_sequence_length)

    stop_at_next_endofsentence = 0
    if decoder_name == 'AttDEC':
        for di in range(max_sequence_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, train = False)
            decoder_attentions[di] = decoder_attention.view(-1, max_sequence_length).data

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
            if ni ==1: 
                if decoder_name == 'AE' and stop_at_next_endofsentence !=1:
                    stop_at_next_endofsentence = 1
                elif decoder_name == 'AE' and stop_at_next_endofsentence == 1:
                    break
                else:
                    break

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([[ni]])).cuda()
        
    if decoder_name == 'AttDEC':
        return decoded_words, decoder_attentions[:di + 1]
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
        loss = train(sentence_1, sentence_2, batch_size, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function)
        print_loss_total += loss
        
        if ((i+1)+epoch*len(train_dataset)) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
            # valid set test
            valid_loss = 0
            for sentence_1, sentence_2 in valid_loader:
                sentence_1 = Variable(sentence_1.view(1, -1)).cuda()
                sentence_2 = Variable(sentence_2.view(1, -1)).cuda()
                loss = train(sentence_1, sentence_2, 1, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, train_data = False)
                valid_loss += loss
                
            if valid_loss < keep_valid_loss:
                keep_valid_loss = valid_loss
                best_valid_epoch = epoch + 1 
                torch.save(encoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d_best_valid_loss.pkl' % (encoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, (epoch+1)))
                torch.save(decoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d_best_valid_loss.pkl' % (decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, (epoch+1)))
                
                print('%s (%d %d%%) loss %.4f, valid loss: %.4f <best valid>' % (timeSince(start, ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) ),
                                         (i+1)+epoch*len(train_dataset), ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) * 100, print_loss_avg, valid_loss))
                
            else:
	            torch.save(encoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d.pkl' % (encoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, (epoch+1)))
	            torch.save(decoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d.pkl' % (decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, (epoch+1)))
	            print('%s (%d %d%%) loss %.4f, valid loss: %.4f' % (timeSince(start, ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) ),
                                         (i+1)+epoch*len(train_dataset), ((i+1)+epoch*len(train_dataset)) / (num_epochs*len(train_dataset)) * 100, print_loss_avg, valid_loss))
            losses = 0


# Test the Model
if not os.path.exists('./result'):
    os.makedirs('./result')
if not os.path.exists('./result/'+directory_name):
    os.makedirs('./result/'+directory_name)

encoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_last_epoch.pkl' % (encoder_name, hidden_size, dropout_rate, teacher_forcing_ratio)))
decoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_last_epoch.pkl' % (decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio)))

with open('./result/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_generation.txt'%(decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio), 'a', encoding ='utf-8') as w:

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
        w.write('Genera_sentence: \t' + ' '.join(predicted)+'\n')
    
    w.write('[setting]: '+'\tbatch_size\t'+str(batch_size)+'\temb_size\t'+str(embedding_size)+'\tHid\t'+str(hidden_size)+'\tD\t'+str(dropout_rate)+'\n')
    w.write('saved to ./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f.pkl\n' % (encoder_name,  hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('saved to ./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f.pkl\n' % (decoder_name,  hidden_size, dropout_rate, teacher_forcing_ratio))
    torch.save(encoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_last_epoch.pkl' % (encoder_name,  hidden_size, dropout_rate, teacher_forcing_ratio))
    torch.save(decoder.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_last_epoch.pkl' % (decoder_name,  hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('-'*100+'\n\n')


# load the best valid model.
#best_valid_epoch = 5
encoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d_best_valid_loss.pkl' % (encoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, best_valid_epoch)))
decoder.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_epoch_%d_best_valid_loss.pkl' % (decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio, best_valid_epoch)))

with open('./result/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_with_best_valid_loss.txt'%(decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio), 'a', encoding ='utf-8') as w:

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
        w.write('Genera_sentence: \t' + ' '.join(predicted)+'\n')
    
    w.write('[setting]: '+'\tbatch_size\t'+str(batch_size)+'\temb_size\t'+str(embedding_size)+'\tHid\t'+str(hidden_size)+'\tD\t'+str(dropout_rate)+'\n')
    w.write('saved to ./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl\n' % (encoder_name, hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('saved to ./models/'+directory_name+'/%s_hid%d_D%0.2f_tfr%0.1f_best_valid_loss.pkl\n' % (decoder_name, hidden_size, dropout_rate, teacher_forcing_ratio))
    w.write('-'*100+'\n\n')
