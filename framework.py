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
        #print "s"
        inp = [self.vcb.stoi[i] for i in sentence.lower().split()] 	#Transforming the sentence to ids
        #print 't'
        embeds = []
        #for i in inp:
        #	embeds.append(ag.Variable(self.vcb.vectors[i]))
        embeds = [ag.Variable(self.vcb.vectors[i]) for i in inp]			#Converting ids to Glove vectors
        #print 'd'
        #print type(embeds)
        #print embeds

        return embeds