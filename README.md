#machine translation with attention
trained a neural machine translation system to translate sentences from English to French.

A brief overview how the model works:

An encoder (LSTM network) encodes the input sequence into a hidden vector z
This vector z serves as the initial hidden state of the decoder (another LSTM network).
The decoder decodes the output sequence starting with a special <s> (start-of-sentence) token until it produces a special </s> (end-of-sentence) token or the maximum length is reached.
The whole system, consisting (in the simplest case) of an encoder, a decoder, and an output projection layer is trained end-to-end on samples of parallel sentences (i.e. a source sentence in english and the corresponding sentence in french).
  
