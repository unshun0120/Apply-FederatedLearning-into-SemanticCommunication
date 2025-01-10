import torch.nn as nn
from semantic_encoder import Encoder
from channel_en_decoder import dense
from model_layer import Channel_VLC
from semantic_decoder import Decoder


class SemanticCommunicationSystem(nn.Module):  # pure DeepSC
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(SemanticCommunicationSystem, self).__init__()
        #self.embedding = embedding(35632, 128)  # which means the corpus has 35632 kinds of words and
        # each word will be coded with a 128 dimensions vector
        # semantic encoder
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        # channel encoder
        self.denseEncoder1 = dense(128, 256)
        self.denseEncoder2 = dense(256, 16)
        # channel decoder
        self.denseDecoder1 = dense(16, 256)
        self.denseDecoder2 = dense(256, 128)
        # semantic decoder
        # TransformerDecoderLayer: made up of self-attn, multi-head-attn and feedforward network
        # TransformerDecoder: a stack of N decoder layers.
        self.decoder = Decoder(enc_shape, Kernel_sz, Nc) 

        self.prediction = nn.Linear(32, 35632)
        self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension

    def forward(self, inputs, snr, cr, channel_type = 'AWGN'):
        #embeddingVector = self.embedding(inputs)
        # semantic encoder
        #code = self.encoder(embeddingVector)
        code = self.encoder(inputs, snr)
        # channel encoder
        denseCode = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(denseCode)
        # AWGN channel 
        #codeWithNoise = AWGN_channel(codeSent, 12)  # assuming snr = 12db
        codeWithNoise = Channel_VLC(codeSent, snr, cr, channel_type)
        # channel decoder
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        # semantic decoder
        codeSemantic = self.decoder(codeReceived, snr)
        codeSemantic = self.prediction(codeSemantic)
        info = self.softmax(codeSemantic)
        
        #return info
        return codeSemantic, inputs


    



