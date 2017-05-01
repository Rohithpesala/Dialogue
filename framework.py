import vocab
from collections import Counter
import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable


class EmbedGlove(nn.Module):
    """
    Component used to get Glove Vector embeddings for sentences
    """

    def __init__(self, corpus, freq):
        """
        Load Data from the corpus.
        Parameters:
        :: corpus :: filepath to read the whole dataset in txt format
		:: freq :: Min frequency of the word to be included into vocab
        """
        super(EmbedGlove, self).__init__()
        f = open(corpus,'r')
        vc = Counter()
        for l in f:
        	vc.update(Counter(l.split()))
        self.vcb = vocab.Vocab(vc, wv_type = "glove.840B",min_freq=freq,specials = ["EOS","SOS"])        

    def forward(self, sentence):
        """
        Parameters:
        :: sentence :: String of the sentence that requires the embedding
        Return:
        :: list of Glove embeddings of each word in the sentence in the same order
        """
        inp = [self.vcb.stoi[i] for i in sentence.lower().split()] 	#Transforming the sentence to ids        
        embeds = [list(self.vcb.vectors[i]) for i in inp]	#Converting ids to Glove vectors        
        #embeds = torch.FloatTensor(embeds)
        return ag.Variable(torch.FloatTensor(embeds))


class EncoderLSTM(nn.Module):
	"""
	Encoder module to encode a given sentence into its meaning that can be fed to the next component
	"""

	def __init__(self, input_size, hidden_size, embedding, dropout, n_layers=1):
		"""
		Parameters:
		:: input_size :: Input size to the LSTM
		:: hidden_size :: Hidden layer size in LSTM
		:: embedding :: The embedding used for the words
		:: dropout :: Dropout parameter for LSTM
		:: n_layers :: The number of LSTM layers in the model		
		"""
		super(EncoderLSTM, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		hidden = self.initHidden()
		self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout = dropout)

	def forward(self, sentence):
		"""
		Parameters:
		:: sentence :: input sentence to be encoded
		Return:
		:: output :: encoded form of the sentence through an LSTM
		"""
		
		embeds = self.embedding(sentence).view(len(sentence.split()),1,-1)	#modify the view to pass to lstm
		#print type(embeds)
		output, self.hidden = self.lstm(embeds, self.hidden)
		return output, self.hidden

	def initHidden(self):
		"""
		Initialize hidden layer		
		"""
		self.hidden = None


class DecoderLSTM(nn.Module):
	"""
	Decoder takes the hidden state and SOS token as input and produces a string ending with EOS token
	"""
	def __init__(self, input_size, hidden_size, output_size, embedding, dropout=0.0, n_layers=1):
		"""
		Parameters:
		:: input_size :: Input size to the LSTM
		:: hidden_size :: Hidden layer size in LSTM
		:: output_size :: The size of last layer where we apply softmax (i.e, vocab size)
		:: embedding :: The embedding used for the words
		:: dropout :: Dropout parameter for LSTM
		:: n_layers :: The number of LSTM layers in the model		
		"""
		super(DecoderLSTM, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		self.hidden = self.initHidden()
		self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout = dropout)
		self.out = nn.Linear(hidden_size,output_size)
		self.sf = nn.Softmax()

	def forward(self, sentence):
		"""
		Parameters:
		:: sentence :: input sentence to be encoded
		Return:
		:: output :: encoded form of the sentence through an LSTM
		"""
		
		embeds = self.embedding(sentence).view(len(sentence.split()),1,-1)	#modify the view to pass to lstm
		#print type(embeds)
		output, self.hidden = self.lstm(embeds, self.hidden)
		output = self.sf(self.out(output.view(len(sentence.split()),-1)))
		return output, self.hidden


	def initHidden(self):
		"""
		Initialize hidden layer		
		"""
		self.hidden = None
		