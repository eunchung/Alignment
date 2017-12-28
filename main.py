import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import sys
import numpy
import os

from utils import *
from models import *

# Hyper Parameters
max_sequence_length = 30
max_vocabulary_size = 25000
embedding_size = 300 #int(sys.argv[6]) #300
hidden_size = int(sys.argv[6])#256
num_layers = 2
num_classes = 2
batch_size = 150
num_epochs = 30
learning_rate = 0.001 #float(sys.argv[6]) #0.003
dropout_rate = float(sys.argv[7]) #0

train_data_path = sys.argv[1]
valid_data_path = sys.argv[2]
test_data_path = sys.argv[3] 
directory_name = sys.argv[4] #'171218'
model_name = sys.argv[5] #'CNN'

word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=False)
#word_to_ix, ix_to_word, vocab_size = make_or_load_dict(train_data_path, character=True)

def model(x):
    return {
        'BiLSTM': BiLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate),
        'CNN': CNN(vocab_size, embedding_size, num_classes, dropout_rate, kernel_num=hidden_size),
        'Cha_CNN_LSTM': Cha_CNN_LSTM(vocab_size, embedding_size, num_classes, dropout_rate, kernel_num =hidden_size),
        'Siamese_BiLSTM': Siamese_BiLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_classes, dropout_rate),
        'Siamese_CNN': Siamese_CNN(vocab_size, num_classes, embedding_size, kernel_num = hidden_size),
    }.get(x)

model = model(model_name)
model.cuda()
#print(model)

if model_name in ['BiLSTM','CNN','Cha_CNN_LSTM'] :
    train_dataset = AlignmentDataset(train_data_path, word_to_ix, batch_size)
    valid_dataset = AlignmentDataset(valid_data_path, word_to_ix, 1)    
    test_dataset = AlignmentDataset(test_data_path, word_to_ix, 1)
elif model_name in ['Siamese_BiLSTM','Siamese_CNN']:
    train_dataset = AlignmentDataset_seperate_sent(train_data_path, word_to_ix, batch_size)
    valid_dataset = AlignmentDataset_seperate_sent(valid_data_path, word_to_ix, 1)
    test_dataset = AlignmentDataset_seperate_sent(test_data_path, word_to_ix, 1)

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset)
test_loader = DataLoader(dataset=test_dataset)
print('length of train dataset', len(train_dataset)*batch_size)
print('length of valid dataset', len(valid_dataset))
print('length of test dataset', len(test_dataset))


# Loss and Optimizer
loss_function = nn.CrossEntropyLoss() # = log softmax + NLL loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print_every = (len(train_dataset)/2)*2
losses = 0
keep_valid_loss = 1e06
keep_valid_accuracy= 0
if not os.path.exists('./models'):
    os.makedirs('./models')
if not os.path.exists('./models/'+directory_name):
    os.makedirs('./models/'+directory_name)

for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    if model_name in ['BiLSTM','CNN','Cha_CNN_LSTM']:
        for i, (sentences, labels) in enumerate(train_loader):
            model.zero_grad()
            logits = model(Variable(sentences.view(batch_size, -1)).cuda())
            loss = loss_function(logits, Variable(labels.view(batch_size)).cuda()) 
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
            losses += loss.data[0]

            if ((i+1)+epoch*len(train_dataset)) % print_every == 0:
                # valid set test
                valid_loss = 0
                valid_correct = 0
                valid_total = 0 
                for sentence, label in valid_loader:
                    output = model(Variable(sentence.view(1, -1)).cuda(), train=False)
                    loss = loss_function(output, Variable(label.view(1)).cuda()) 
                    valid_loss += loss.data[0]

                    _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환
                    valid_total += label.size(0)
                    valid_correct += (predicted.cpu() == label).sum()
                valid_accuracy = (100 * valid_correct / valid_total)
                
                # test set test
                test_loss = 0
                test_correct = 0
                test_total = 0 
                for sentence, label in test_loader:
                    output = model(Variable(sentence.view(1, -1)).cuda(), train=False)
                    loss = loss_function(output, Variable(label.view(1)).cuda()) 
                    test_loss += loss.data[0]

                    _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환
                    test_total += label.size(0)
                    test_correct += (predicted.cpu() == label).sum()
                test_accuracy = (100 * test_correct / test_total)

                if valid_accuracy > keep_valid_accuracy:
                    keep_valid_accuracy = valid_accuracy
                # if valid_loss < keep_valid_loss:
                #     keep_valid_loss = valid_loss
                    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_epoch_%d_best_valid_accuracy.pkl' % (model_name, hidden_size, dropout_rate, (epoch+1)))
                    print ('Ep [%d/%d], Step [%d/%d], L: %.4f, V_L: %.4f, V_ACC: %0.2f %%, Te_L: %.4f, Te_ACC: %0.2f %% <best valid>' 
                       %(epoch+1, num_epochs, (i+1)+epoch*len(train_dataset), num_epochs*len(train_dataset), losses, valid_loss, valid_accuracy, test_loss, test_accuracy))   

                else:
                    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_epoch_%d.pkl' % (model_name, hidden_size, dropout_rate, (epoch+1)))
                    print ('Ep [%d/%d], Step [%d/%d], L: %.4f, V_L: %.4f, V_ACC: %0.2f %%, Te_L: %.4f, Te_ACC: %0.2f %%' 
                       %(epoch+1, num_epochs, (i+1)+epoch*len(train_dataset), num_epochs*len(train_dataset), losses, valid_loss, valid_accuracy, test_loss, test_accuracy))
                losses = 0


    elif model_name in ['Siamese_BiLSTM','Siamese_CNN']:
        for i, (sentence_1, sentence_2, labels) in enumerate(train_loader):
            model.zero_grad()
            logits = model(Variable(sentence_1.view(batch_size, -1)).cuda(), Variable(sentence_2.view(batch_size, -1)).cuda()) 
            loss = loss_function(logits, Variable(labels.view(batch_size)).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
            losses += loss.data[0]

            if ((i+1)+epoch*len(train_dataset)) % print_every == 0:
                # valid set test
                valid_loss = 0
                valid_correct = 0
                valid_total = 0 
                for sentence_1, sentence_2, label in valid_loader:
                    output = model(Variable(sentence_1.view(1, -1)).cuda(), Variable(sentence_2.view(1, -1)).cuda(), train=False)
                    loss = loss_function(output, Variable(label.view(1)).cuda()) 
                    valid_loss += loss.data[0]

                    _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환
                    valid_total += label.size(0)
                    valid_correct += (predicted.cpu() == label).sum()
                valid_accuracy = (100 * valid_correct / valid_total)

                # test set test
                test_loss = 0
                test_correct = 0
                test_total = 0 
                for sentence_1, sentence_2, label in test_loader:
                    output = model(Variable(sentence_1.view(1, -1)).cuda(), Variable(sentence_2.view(1, -1)).cuda(), train=False)
                    loss = loss_function(output, Variable(label.view(1)).cuda()) 
                    test_loss += loss.data[0]

                    _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환
                    test_total += label.size(0)
                    test_correct += (predicted.cpu() == label).sum()
                test_accuracy = (100 * test_correct / test_total)

                if valid_accuracy > keep_valid_accuracy:
                    keep_valid_accuracy = valid_accuracy
                # if valid_loss < keep_valid_loss:
                #     keep_valid_loss = valid_loss
                    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_epoch_%d_best_valid_accuracy.pkl' % (model_name, hidden_size, dropout_rate, (epoch+1)))
                    print ('Ep [%d/%d], Step [%d/%d], L: %.4f, V_L: %.4f, V_ACC: %0.2f %%, Te_L: %.4f, Te_ACC: %0.2f %% <best valid>' 
                       %(epoch+1, num_epochs, (i+1)+epoch*len(train_dataset), num_epochs*len(train_dataset), losses, valid_loss, valid_accuracy, test_loss, test_accuracy))   

                else:
                    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_epoch_%d.pkl' % (model_name, hidden_size, dropout_rate, (epoch+1)))
                    print ('Ep [%d/%d], Step [%d/%d], L: %.4f, V_L: %.4f, V_ACC: %0.2f %%, Te_L: %.4f, Te_ACC: %0.2f %%' 
                       %(epoch+1, num_epochs, (i+1)+epoch*len(train_dataset), num_epochs*len(train_dataset), losses, valid_loss, valid_accuracy, test_loss, test_accuracy))
                losses = 0


# Test the Model
correct = 0
total = 0
missed_pairs = []
if model_name in ['BiLSTM','CNN','Cha_CNN_LSTM']:
    for sentence, label in test_loader:

        outputs = model(Variable(sentence.view(1, -1)).cuda(), train=False)
        _, predicted = torch.max(outputs.data, 1) # 두번째 아웃풋 값은 argmax 를 반환

        total += label.size(0) # batch 쓰는 경우.
        if (predicted.cpu().numpy() != label.numpy()):
            sentence = [ix_to_word[word_idx] for word_idx in sentence.long()[0][0].numpy()]
            missed_pairs.append('label: ' + str(label.numpy()[0][0]) +'\t' + 'predicted: '+str(predicted.cpu().numpy()) + '\t' +  ' '.join(sentence)+'\n')
        correct += (predicted.cpu() == label).sum()

elif model_name in ['Siamese_BiLSTM','Siamese_CNN']:
    for sentence_1, sentence_2, label in test_loader:

        output = model(Variable(sentence_1.view(1, -1)).cuda(), Variable(sentence_2.view(1, -1)).cuda(), train=False)
        _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환

        total += label.size(0) # batch 쓰는 경우.
        if (predicted.cpu().numpy() != label.numpy()):
            sentence_1 = [ix_to_word[word_idx] for word_idx in sentence_1.long()[0][0].numpy()]
            sentence_2 = [ix_to_word[word_idx] for word_idx in sentence_2.long()[0][0].numpy()]
            missed_pairs.append('label: ' + str(label.numpy()[0][0]) +'\t' + 'predicted: '+str(predicted.cpu().numpy()) + '\t' + ' '.join(sentence_1)+'\t'+' '.join(sentence_2)+'\n')
        correct += (predicted.cpu() == label).sum()

print('Test Accuracy of the model: %0.2f %%' % (100 * correct / total))
# write result and save model
if not os.path.exists('./result'):
    os.makedirs('./result')
if not os.path.exists('./result/'+directory_name):
    os.makedirs('./result/'+directory_name)

with open('./result/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f.txt' % (model_name, hidden_size, dropout_rate, 100 * correct / total), 'a', encoding ='utf-8') as w:

    w.write('[setting]: '+'\tbatch_size\t'+str(batch_size)+'\temb_size\t'+str(embedding_size)+'\tHid\t'+str(hidden_size)+'\tD\t'+str(dropout_rate)+'\n')
    w.write('[Test Accuracy of the model]: %0.2f %% \n' % (100 * correct / total)) 
    w.write('[saved to]: ./models/%s_hid%d_D%0.2f_Acc%0.2f.pkl\n' % (model_name, hidden_size, dropout_rate, 100 * correct / total))
    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f_last_epoch.pkl' % (model_name, hidden_size, dropout_rate, 100 * correct / total))
    
    for miss in missed_pairs:
        w.write(miss)
    w.write('-'*100+'\n\n')


# load the best valid model.
correct = 0
total = 0
missed_pairs = []
best_valid_epoch = 10
model.load_state_dict(torch.load('./models/'+directory_name+'/%s_hid%d_D%0.2f_epoch_%d_best_valid_accuracy.pkl' % (model_name, hidden_size, dropout_rate, best_valid_epoch)))
if model_name in ['BiLSTM','CNN','Cha_CNN_LSTM']:
    for sentence, label in test_loader:

        outputs = model(Variable(sentence.view(1, -1)).cuda(), train=False)
        _, predicted = torch.max(outputs.data, 1) # 두번째 아웃풋 값은 argmax 를 반환

        total += label.size(0) # batch 쓰는 경우.
        if (predicted.cpu().numpy() != label.numpy()):
            sentence = [ix_to_word[word_idx] for word_idx in sentence.long()[0][0].numpy()]
            missed_pairs.append('label: ' + str(label.numpy()[0][0]) +'\t' + 'predicted: '+str(predicted.cpu().numpy()) + '\t' +  ' '.join(sentence)+'\n')
        correct += (predicted.cpu() == label).sum()

elif model_name in ['Siamese_BiLSTM','Siamese_CNN']:
    for sentence_1, sentence_2, label in test_loader:

        output = model(Variable(sentence_1.view(1, -1)).cuda(), Variable(sentence_2.view(1, -1)).cuda(), train=False)
        _, predicted = torch.max(output.data, 1) # 두번째 아웃풋 값은 argmax 를 반환

        total += label.size(0) # batch 쓰는 경우.
        if (predicted.cpu().numpy() != label.numpy()):
            sentence_1 = [ix_to_word[word_idx] for word_idx in sentence_1.long()[0][0].numpy()]
            sentence_2 = [ix_to_word[word_idx] for word_idx in sentence_2.long()[0][0].numpy()]
            missed_pairs.append('label: ' + str(label.numpy()[0][0]) +'\t' + 'predicted: '+str(predicted.cpu().numpy()) + '\t' + ' '.join(sentence_1)+'\t'+' '.join(sentence_2)+'\n')
        correct += (predicted.cpu() == label).sum()

print('Test Accuracy of the best valid accuracy model: %0.2f %%' % (100 * correct / total))
# write result
with open('./result/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f_best_valid_accuracy.txt' % (model_name, hidden_size, dropout_rate, 100 * correct / total), 'a', encoding ='utf-8') as w:

    w.write('[setting]: '+'\tbatch_size\t'+str(batch_size)+'\temb_size\t'+str(embedding_size)+'\tHid\t'+str(hidden_size)+'\tD\t'+str(dropout_rate)+'\n')
    w.write('[Test Accuracy of the model]: %0.2f %% \n' % (100 * correct / total)) 
    w.write('[saved to]: ./models/%s_hid%d_D%0.2f_Acc%0.2f_best_valid_accuracy.pkl\n' % (model_name, hidden_size, dropout_rate, 100 * correct / total))
    torch.save(model.state_dict(), './models/'+directory_name+'/%s_hid%d_D%0.2f_Acc%0.2f_best_valid_accuracy.pkl' % (model_name, hidden_size, dropout_rate, 100 * correct / total))
    
    for miss in missed_pairs:
        w.write(miss)
    w.write('-'*100+'\n\n')