########################################################
############### Libraries import #######################
########################################################
import numpy as np
#np.random.bit_generator = np.random._bit_generator
import pandas as pd
import torch
from torch import nn
import gensim
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


########################################################
############ Check Device for GPU ######################
########################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################################
#################### Data Loading ######################
########################################################
data = pd.read_csv("tripadvisor_hotel_reviews.csv")
data_x = data['Review']
data_y = data['Rating']
data_y = np.array(data_y.values) - 1
data_y = data_y



########################################################
########### Google word2vec model Loading ##############
########################################################
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
 
 

########################################################
####### Data Split in Train, Validation & Test #########
########################################################
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2 ( 20% for validation dataset)

# Push the data to the GPU
y_train = torch.tensor(y_train).long()
y_train = y_train.to(device)

y_test = torch.tensor(y_test).long()
y_test = y_test.to(device)

y_val = torch.tensor(y_val).long()
y_val = y_test.to(device)


########################################################
#################### RNN Model #########################
########################################################
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nb_layers):
        super(RNN, self).__init__()

        # Number of units per hidden layer
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        
        # Our RNN Layer, with  batch_first=True to specify to the input tensors are provided as (batch, seq, feature)
        self.rnn = nn.RNN(input_size, hidden_size, nb_layers, batch_first=True)   
        # Output layer
        self.linear = nn.Linear(hidden_size, output_size)
    
    ################### Forward function ##################
    def forward(self, x):
        
        # Number of examples feeded to the RNN network
        batch_size = len(x)
        
        # Initializing the first hidden state 
        hidden = self.init_hidden(batch_size)
        
        # Embedding layer
        x = self.embedding_layer(x,batch_size).to(device)
        
        # Feeding the input and hidden state into the model and obtaining outputs
        output, hidden = self.rnn(x, hidden)
        output = output[:,-1,:]
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.linear(output)
        
        
        return output, hidden
        
        ##################### Embedding Layer #####################
    def embedding_layer(self, x, batch_size):
        
        # Deleting some stop words
        df = x.str.replace(',', '')
        df = df.str.replace('.', '')
        df = df.str.replace('!', '')
        df = df.str.replace('?', '')
        
        # Split each input text in a vector of words
        df = df.str.split() 
        
        # Checking the longest sequence to apply padding to the other sequences
        max_seq_length = 0
        for k in range(batch_size):
            l = len(list(df)[k])
            if max_seq_length < l:
                max_seq_length = l
                
        # For each input, generate its word vectors
        embedded_data = []
        for k in range(batch_size):
            my_list = list(df)[k]
            seq_length = len(my_list) 
            my_list = my_list + ['padding']*(max_seq_length - seq_length) # Adding word 'padding' if necessary
            
               
            seq = []
            for m in range(max_seq_length):
               
                try:
                    temp = model[my_list[m]] # Generate word vector of a word in the sequence
                    seq.append(temp)
                except KeyError: # If the word is not in the model Vocabulary
                    if my_list[m].isdigit():
                        seq.append(model['number']) # Replace it by the word vector 'number' if the word is a number
                    else:
                        seq.append(model['unknown']) # Replace it by the word vector 'unknown' if the word is not a number
                        
            embedded_data.append(seq)
            
        return torch.tensor(embedded_data).float() # Pushing the matrix of word vector to the device
        
        
        
    
    ############### Initialization of the initial hidden state ############################
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.nb_layers, batch_size, self.hidden_size).to(device)
        return hidden
        
 

########################################################
################# Learning Parameter ###################
########################################################
 
EPOCHS = 10
BATCH_SIZE_TRAIN = 1300
BATCH_SIZE_VALID = 1
BATCH_SIZE_TEST = 1
LEARNING_RATE = 0.05
NUMB_FEATURES = 300
HIDDEN_LAYER_SIZE = 12
NB_LAYERS = 2
NUMB_CLASSES = 5


########################################################
############### Loss & Accuracy Tracking ###############
########################################################
accuracy_stats = {
    'train': [],
    "val": []
}

loss_stats = {
    'train': [],
    "val": []
}




########################################################
################## Model Creation ######################
########################################################
model_rnn = RNN(NUMB_FEATURES, NUMB_CLASSES, HIDDEN_LAYER_SIZE, NB_LAYERS)
model_rnn.to(device) # Pushing the model to the device
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_rnn.parameters(), lr = LEARNING_RATE, )



########################################################
################ Accuracy Computation ##################
########################################################
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1) # Applying the logSoftmax on the linear output 
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) # Conversion of the results into classes
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc
    


########################################################
#################### Batch Loader ######################
########################################################
def batch_loader(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]






    

########################################################
############# Model Training & Validation ##############
########################################################
print("Begin training.")
best_accuracy = 0
for e in range(1, EPOCHS+1):
    
    # TRAINING
    nb_train_chunk = 0
    train_epoch_loss = 0
    train_epoch_acc = 0
    model_rnn.train()
    for X_train_batch, y_train_batch in zip(batch_loader(X_train, BATCH_SIZE_TRAIN),batch_loader(y_train, BATCH_SIZE_TRAIN)):
        nb_train_chunk += 1
        optimizer.zero_grad()
        
        
        y_train_pred, _ = model_rnn(X_train_batch)
                
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model_rnn.eval()
        nb_val_chunk = 0
        for X_val_batch, y_val_batch in zip(batch_loader(X_val, BATCH_SIZE_VALID),batch_loader(y_val, BATCH_SIZE_VALID)):
            nb_val_chunk += 1  
            
            y_val_pred, _ = model_rnn(X_val_batch) 
          
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            
    # Saving the best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model_rnn.state_dict(), 'best-model.pt')
        
    loss_stats['train'].append(train_epoch_loss/nb_train_chunk)
    loss_stats['val'].append(val_epoch_loss/nb_val_chunk)
    accuracy_stats['train'].append(train_epoch_acc/nb_train_chunk)
    accuracy_stats['val'].append(val_epoch_acc/nb_val_chunk)
                              
    
    print('Epoch : ', e ,' | Train Loss: ', round(train_epoch_loss/nb_train_chunk, 3) ,' | Val Loss: ', round(val_epoch_loss/nb_val_chunk, 3) , ' | Train Acc: ', round(train_epoch_acc/nb_train_chunk,3) ,' | Val Acc: ', round(val_epoch_acc/nb_val_chunk,3) )
    
    
# TEST

# Loading the best model   
best_model = RNN(NUMB_FEATURES, NUMB_CLASSES, HIDDEN_LAYER_SIZE, NB_LAYERS)
best_model.load_state_dict(torch.load('best-model.pt'))
best_model.to(device)

y_pred_list = []
accuracy = 0
with torch.no_grad():
    best_model.eval()
    nb_test_chunk = 0
    for X_batch, _ in zip(batch_loader(X_test, BATCH_SIZE_TEST),batch_loader(y_test, BATCH_SIZE_TEST)):
        nb_test_chunk += 1 
        
        y_test_pred, _ = best_model(X_batch)
        accuracy += multi_acc(y_test_pred, y_test)
        


print('Test accuracy :')
print(round(accuracy/len(y_test),3))