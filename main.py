import vocab
import os
import framework as fw
import constants as const
from collections import Counter
import torch
import torch.nn as nn
import torch.autograd as ag
from torch import optim
from torch.autograd import Variable
import time
import math


#######################################################################################
#Adopted and modified from pytorch tutorials

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#######################################################################################

def train_lstm_instance(input_sentence, target_sentence, encoder, decoder, encoder_opt, decoder_opt, crit, embedding):
	"""
	Parameters:
	::
	Return:
	:: Loss
	"""
	#initializing values
	encoder.initHidden()
	encoder_opt.zero_grad()
	decoder_opt.zero_grad()
	loss = 0
	
	#Encoder Layer
	inp_encoder = embedding(input_sentence).cuda()
	tmp, h = encoder(inp_encoder)

	#Decoder Layer
	inp_decoder = const.SOS_Token + " " + target_sentence
	tar_decoder = target_sentence + " " + const.EOS_Token
	inp_embed = embedding(inp_decoder).cuda()
	tar_indexes = embedding.generateID(tar_decoder).cuda()
	#print inp_embed[1]
	decoder.hidden = h
	#decoder.initHidden()
	
	out, h = decoder(inp_embed)
	out1 = out.clone()
	#print out1[1]
	loss+=crit(out,ag.Variable(torch.LongTensor(tar_indexes)))
	loss.backward()
	encoder_opt.step()
	decoder_opt.step()
	return loss.data[0]


def train_lstm(corpus,n_epochs=1, save_every=100,freq=10,min_len_sentence = 10,n_layers=3,dropout = 0.1):
	start = time.time()
	embedding = fw.EmbedGlove(corpus,freq)
	vocab_size = len(embedding.vcb)
	embed_dim = 300
	encoder = fw.EncoderLSTM(embed_dim,embed_dim*2,dropout,n_layers=n_layers).cuda()
	decoder = fw.DecoderLSTM(embed_dim,embed_dim*2,vocab_size,n_layers=n_layers).cuda()
	if os.path.isfile(os.getcwd()+"/Checkpoints/encoder"):
		encoder = torch.load(os.getcwd()+"/Checkpoints/encoder")
	if os.path.isfile(os.getcwd()+"/Checkpoints/decoder"):
		decoder = torch.load(os.getcwd()+"/Checkpoints/decoder")
	encoder_opt = optim.SGD(encoder.parameters(),lr = 0.01)
	decoder_opt = optim.SGD(decoder.parameters(),lr = 0.01)
	tot_loss = 0.0
	f = open(corpus,'r')
	i=0
	prev = ""
	for l in f:
		i+=1
		if i%save_every == 0:
			print "==================================================================================================================="
			print "Step = ",i			
			print "Loss = ",tot_loss/save_every
			print "Time = ",timeSince(start,i/87980.0)
			tot_loss = 0
			torch.save(encoder,os.getcwd()+"/Checkpoints/encoder")
			torch.save(decoder,os.getcwd()+"/Checkpoints/decoder")
		if i == 1:
			prev = l
			continue
		pres = l
		if len(pres.split())>min_len_sentence or len(prev.split())>min_len_sentence:
			prev = l
			continue
		else:
			tot_loss+=train_lstm_instance(prev,pres,encoder,decoder,encoder_opt,decoder_opt,nn.NLLLoss(),embedding)


def evaluate(input, target, encoder, decoder, embedding, mod=None):
	pass