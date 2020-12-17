import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers  
        
        # Embedded layer
        self.embedding_nn = nn.Embedding(vocab_size, embed_size)
        
        #LSTM declaration
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first= True)
        
        #fully connected layer declaration
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        
        
        
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device = device),
                torch.zeros(1, batch_size, self.hidden_size, device = device))
    
    def forward(self, features, captions):
        
        #batch size
        self.batch_size=features.shape[0]
        
        #initialize the hidden LSTM layer with zeros
        self.hidden_layer_lstm=self.init_hidden(self.batch_size)
        
        #Truncating the captions. We throw away the <END> world because there is no world after it that the network predicts!
        #Then we propagate each of the vectors representing words through NN and get its 256x1 vector representation. After that
        #we combine them into one 256x(caption_length-1) vector.
        embedded_layer = self.embedding_nn(captions[:,:-1])
        
        #Vector features represents 256x1 image feature vector. Vector embedded_layer represents 256x(caption_length-1) vector consisting of
        #256x1 vectors (there are caption_length-1 of them) representing each caption word. Since these vectors are used 
        #for every batch_size images, their sizes would be batch_sizex256x1 and batch_sizex256xcaptions_length, respectively. 
        #Then we merge these two vectors. Image feature vector goes first!!!
        
        lstm_input = torch.cat((features.unsqueeze(1), embedded_layer), dim=1)
        
        # LSTM propogation
        lstm_output, self.hidden_layer_lstm = self.lstm(lstm_input, self.hidden_layer_lstm)
        
        #return result after final propagation through fully connected layer
        return self.fc(lstm_output)
        

    def sample(self, inputs, states=None, max_len=20):
        
        '''Caption generation'''
        
        #initialize the hidden LSTM layer with zeros
        self.hidden_layer_lstm=self.init_hidden(self.batch_size)
        
        #resulting word vector 
        word_list=[]
        
        #Since there are no captions during test, self.embedding_nn is not used! We propogate word vectors one by one through LSTM
        #using previous word vector as as input.
        
        with torch.no_grad(): # no gradient propagation
            for i in range(max_len):
                lstm_output, self.hidden_layer_lstm = self.lstm(inputs, self.hidden_layer_lstm)
                
                #get the word number as the maximum value index
                word = self.fc(lstm_output).squeeze(1).argmax(dim=1)
                word_list.append(word)
                
                #if <END> is reached then break the cycle
                if word == 1:
                    break
                    
        return word_list
        
        
        
        