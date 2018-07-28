
import numpy as np
import torch
import torch.nn.functional as F

import torch.utils.data

from utils import get_sequences_lengths, variable, argmax, cuda, get_sentence_from_indices


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, padding_idx, init_idx, max_len, teacher_forcing):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing

        self.embedding = torch.nn.Embedding (vocab_size, embedding_dim)
        self.enc_LSTM = torch.nn.LSTM(embedding_dim, hidden_size)
        self.attn = torch.nn.Linear(hidden_size * 2, max_len)
        self.attn_combine = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.dec_embedding = torch.nn.Embedding (vocab_size, hidden_size)
        self.dec_LSTM = torch.nn.LSTMCell(hidden_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=1)



    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1
        state_shape = (nb_layers, batch_size, self.hidden_size)

        h = variable(torch.zeros(state_shape))
        c = variable(torch.zeros(state_shape))
        h = cuda(h)
        c = cuda(c)

        
        return h, c

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """

        batch_size = inputs.size(0)
        
        
        outputs = []
        hidden =  self.zero_state(batch_size)
        for i in range(inputs.size(1)):
            inp = inputs.select(1,i)
            embed = self.embedding(inp).view(1, batch_size,-1)
            #print (embed.size())
            output, hidden = self.enc_LSTM(embed, hidden)
            outputs.append(output)
        outputs = torch.cat(outputs)
        z = hidden[0][0]
        
        
            
            
        
        return z, outputs

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)
        #print (batch_size)

        state_shape = (batch_size, self.hidden_size)
        c0 = variable(torch.zeros(state_shape))
        c0 = cuda(c0)
        
        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        inputs = variable(np.full((1,), self.init_idx)).expand((batch_size,))
        return inputs

    def decode_sentence(self, z, enc_output, targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = z.size(0)  
        
        hidden = self.decoder_state(z)
        inputs = self.decoder_initial_inputs (batch_size)
        outputs = []     
        for i in range (self.max_len):
            embed = self.dec_embedding(inputs)

            attn = self.attn(torch.cat((embed, hidden[0]), 1))
            attn_weights  =  F.softmax( attn, dim = 1)

            attn_applied = torch.bmm(attn_weights.unsqueeze(0).transpose(0,1),
                                 enc_output.transpose(0,1))
            attn_applied = attn_applied.transpose(0,1)
            
            output = torch.cat((embed, attn_applied[0]), 1)
            output = self.attn_combine(output)
            output = F.relu(output)
            
            hidden = self.dec_LSTM (output, hidden)
            output = self.linear(hidden[0])
              
            if targets is None:
                #inputs = argmax(output).squeeze(0)
                inputs = self.softmax(output)
                inputs = torch.multinomial(inputs, 1) 
                inputs = inputs.transpose(0,1).squeeze(0)
                
            else:
                inputs = targets.select(1,i)

                 
            outputs.append(output)
        outputs = torch.cat(outputs)
        

        return outputs

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        if self.training and np.random.rand() < self.teacher_forcing:
            targets = targets
        else:
            targets = None

        z, enc_output = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, enc_output, targets)
        return outputs