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

###############################################################################################################
#Seq2Seq Lstm model
###############################################################################################################

def train_lstm_instance(input_sentence, target_sentence, encoder, decoder, encoder_opt, decoder_opt, crit, embedding):
	"""
	Parameters:
	::
	Return:
	:: Loss
	"""
	#initializing values
	encoder.initHidden()
	# encoder_opt.zero_grad()
	# decoder_opt.zero_grad()
	loss = 0
	
	#Encoder Layer
	inp_encoder = embedding(input_sentence+ " " + const.EOS_Token)#.cuda()
	tmp, h = encoder(inp_encoder)
	# print inp_encoder.requires_grad
	# print h[1].view(1,-1)

	#Decoder Layer
	inp_decoder = const.SOS_Token + " " + target_sentence
	tar_decoder = target_sentence + " " + const.EOS_Token
	inp_embed = embedding(inp_decoder)#.cuda()
	tar_indexes = embedding.generateID(tar_decoder)#.cuda()
	#print inp_embed[1]
	decoder.hidden = h
	#decoder.initHidden()
	
	out, h = decoder(inp_embed)
	out1 = out.clone()
	#print out1[1]
	loss+=crit(out,ag.Variable(torch.LongTensor(tar_indexes)))
	# loss.backward()
	# encoder_opt.step()
	# decoder_opt.step()
	return loss


def train_lstm(corpus,n_epochs=1, save_every=10000,freq=10,min_len_sentence = 10,n_layers=3,dropout = 0.1,batch_size=64):
	start = time.time()
	embedding = fw.EmbedGlove(corpus,freq)
	vocab_size = len(embedding.vcb)
	embed_dim = 300
	hid_dim = embed_dim/2
	encoder = fw.EncoderLSTM(embed_dim,hid_dim,dropout,n_layers=n_layers)#.cuda()
	decoder = fw.DecoderLSTM(embed_dim,hid_dim,vocab_size,n_layers=n_layers)#.cuda()
	print hid_dim
	print vocab_size
	if os.path.isfile(os.getcwd()+"/Checkpoints/encoder"):
		encoder = torch.load(os.getcwd()+"/Checkpoints/encoder")
	if os.path.isfile(os.getcwd()+"/Checkpoints/decoder"):
		decoder = torch.load(os.getcwd()+"/Checkpoints/decoder")
	encoder_opt = optim.SGD(encoder.parameters(),lr = 0.01, momentum = 0.1)
	decoder_opt = optim.SGD(decoder.parameters(),lr = 0.01, momentum = 0.1)
	encoder_opt.zero_grad()
	decoder_opt.zero_grad()
	loss = 0
	for ep in range(n_epochs):
		tot_loss = 0.0
		f = open(corpus,'r')
		i=0
		prev = ""
		# flist = [0 for i in range(90000)]
		# i=0
		# for l in f:
		# 	flist[i] = l
		# 	i+=1
		# i=0
		loss_f = open("Seq_loss.txt","w")
		for l in f:
			i+=1
			if i%save_every == 0:
				print "==================================================================================================================="
				print "Epoch = ",ep
				print "Step = ",i			
				print "Loss = ",tot_loss/save_every
				print "Time = ",timeSince(start,i/10000.0)
				tot_loss = 0
				torch.save(encoder,os.getcwd()+"/Checkpoints/encoder")
				torch.save(decoder,os.getcwd()+"/Checkpoints/decoder")
			if i == 1:
				prev = l
				continue
			if i%batch_size == 0:
				encoder_opt.step()
				decoder_opt.step()
				encoder_opt.zero_grad()
				decoder_opt.zero_grad()
			pres = l
			if len(pres.split())>min_len_sentence or len(prev.split())>min_len_sentence:
				prev = l
				continue
			else:
				loss = 0
				loss =train_lstm_instance(prev,pres,encoder,decoder,encoder_opt,decoder_opt,nn.CrossEntropyLoss(),embedding)
				tot_loss+=loss.data[0]
				loss.backward()
		f.close()

def predict_lstm(input_sentence,corpus,freq):
	embedding = fw.EmbedGlove(corpus,freq)
	vocab_size = len(embedding.vcb)
	embed_dim = 300
	# encoder = fw.EncoderLSTM(embed_dim,embed_dim,dropout,n_layers=n_layers).cuda()
	# decoder = fw.DecoderLSTM(embed_dim,embed_dim,vocab_size,n_layers=n_layers).cuda()
	if os.path.isfile(os.getcwd()+"/Checkpoints/encoder"):
		encoder = torch.load(os.getcwd()+"/Checkpoints/encoder")
	else:
		print "No encoder present"
		return 0
	if os.path.isfile(os.getcwd()+"/Checkpoints/decoder"):
		decoder = torch.load(os.getcwd()+"/Checkpoints/decoder")
	else:
		print "No decoder present"
		return 0
	
	while 1:
		input_sentence = str(raw_input())
		encoder.initHidden()
		#Encoder Layer
		inp_encoder = embedding(input_sentence+ " " + const.EOS_Token)#.cuda()
		tmp, h = encoder(inp_encoder)
		# print inp_encoder

		#Decoder Layer
		#inp_decoder = const.SOS_Token + " " + target_sentence
		#tar_decoder = target_sentence + " " + const.EOS_Token
		inp_embed = embedding(const.SOS_Token)#.cuda()
		#tar_indexes = embedding.generateID(tar_decoder)#.cuda()
		#print inp_embed[1]
		decoder.hidden = h
		#decoder.initHidden()
		pword = ""
		out_str = ""
		while pword!=const.EOS_Token:
			out_str += pword + " "
			print pword
			out, h = decoder(inp_embed)
			# print out.clone()
			ind = out.data.topk(1)[1][0][0]
			pword = embedding.itoword(ind)
			inp_embed = embedding(pword)
		print out_str

####################################################################################################################
# Mem network
# ##################################################################################################################

def train_mem_instance(n_layers, input_sentence, target_sentence, context_sentences, encoder, decoder, mem_network, memory, encoder_opt, decoder_opt, crit, embedding):
	"""
	Parameters:
	::
	Return:
	:: Loss
	"""
	#initializing values
	encoder.initHidden()
	# encoder_opt.zero_grad()
	# decoder_opt.zero_grad()
	loss = 0
	
	#Encoder Layer
	inp_encoder = embedding(input_sentence+ " " + const.EOS_Token)#.cuda()
	tmp, hq = encoder(inp_encoder)


	mem_mat = 0
	for i in range(memory):
		inp_encoder = embedding(context_sentences[i]+ " " + const.EOS_Token)
		tmp, h = encoder(inp_encoder)
		if mem_mat==0:
			mem_mat = h[1].view(1,-1)
		else:
			mem_mat = torch.cat((mem_mat,h[1].view(1,-1)))
	mem_mat = torch.t(mem_mat)

	c = mem_network(hq[1].view(1,-1),mem_mat)

	hm = (hq[0],c.view(n_layers,1,-1))

	#Decoder Layer
	inp_decoder = const.SOS_Token + " " + target_sentence
	tar_decoder = target_sentence + " " + const.EOS_Token
	inp_embed = embedding(inp_decoder)#.cuda()
	tar_indexes = embedding.generateID(tar_decoder)#.cuda()
	#print inp_embed[1]
	decoder.hidden = hm
	#decoder.initHidden()
	
	out, h = decoder(inp_embed)
	out1 = out.clone()
	#print out1[1]
	loss+=crit(out,ag.Variable(torch.LongTensor(tar_indexes)))
	# loss.backward()
	# encoder_opt.step()
	# decoder_opt.step()
	return loss

def train_mem(corpus,n_epochs=1, save_every=100,freq=10,min_len_sentence = 10,n_layers=3,dropout = 0.1,batch_size=64,memory=5,n_hops=1):
	start = time.time()
	embedding = fw.EmbedGlove(corpus,freq)
	vocab_size = len(embedding.vcb)
	embed_dim = 300
	hid_dim = embed_dim/2
	encoder = fw.EncoderLSTM(embed_dim,hid_dim,dropout,n_layers=n_layers)#.cuda()
	decoder = fw.DecoderLSTM(embed_dim,hid_dim,vocab_size,n_layers=n_layers)#.cuda()
	mem_dim = n_layers*hid_dim
	mem_network = fw.Mem_net(mem_dim,memory,n_hops)
	print hid_dim
	print vocab_size
	if os.path.isfile(os.getcwd()+"/Checkpoints/encoder"):
		encoder = torch.load(os.getcwd()+"/Checkpoints/encoder_mem")
	if os.path.isfile(os.getcwd()+"/Checkpoints/decoder"):
		decoder = torch.load(os.getcwd()+"/Checkpoints/decoder_mem")
	encoder_opt = optim.SGD(encoder.parameters(),lr = 0.01, momentum = 0.1)
	decoder_opt = optim.SGD(decoder.parameters(),lr = 0.01, momentum = 0.1)
	encoder_opt.zero_grad()
	decoder_opt.zero_grad()
	loss = 0
	loss_f = open("mem_loss.txt","w")
	for ep in range(n_epochs):
		tot_loss = 0.0
		f = open(corpus,'r')
		i=0
		prev = ""
		# flist = [0 for i in range(90000)]
		# i=0
		# for l in f:
		# 	flist[i] = l
		# 	i+=1
		# i=0
		mem_sent = [const.SOS_Token for i in range(memory)]
		for l in f:
			i+=1
			if i%save_every == 0:
				print "==================================================================================================================="
				print "Epoch = ",ep
				print "Step = ",i			
				print "Loss = ",tot_loss/save_every
				print "Time = ",timeSince(start,i/10000.0)
				loss_f.write(str(tot_loss/save_every) + "\n")
				tot_loss = 0
				print mem_sent
				torch.save(encoder,os.getcwd()+"/Checkpoints/encoder_mem")
				torch.save(decoder,os.getcwd()+"/Checkpoints/decoder_mem")
			if i == 1:
				prev = l
				continue
			if i%batch_size == 0:
				encoder_opt.step()
				decoder_opt.step()
				encoder_opt.zero_grad()
				decoder_opt.zero_grad()
			pres = l
			if len(pres.split())>min_len_sentence or len(prev.split())>min_len_sentence:
				prev = l
				continue
			else:
				loss = 0
				loss =train_mem_instance(n_layers,prev,pres,mem_sent,encoder,decoder,mem_network,memory,encoder_opt,decoder_opt,nn.CrossEntropyLoss(),embedding)
				tot_loss+=loss.data[0]
				loss.backward()
				mem_sent = mem_sent[1:memory] + [l]
		f.close()
	loss_f.close()

def predict_mem(corpus,freq):
	embedding = fw.EmbedGlove(corpus,freq)
	vocab_size = len(embedding.vcb)
	embed_dim = 300
	# encoder = fw.EncoderLSTM(embed_dim,embed_dim,dropout,n_layers=n_layers).cuda()
	# decoder = fw.DecoderLSTM(embed_dim,embed_dim,vocab_size,n_layers=n_layers).cuda()
	if os.path.isfile(os.getcwd()+"/Checkpoints/encoder"):
		encoder = torch.load(os.getcwd()+"/Checkpoints/encoder")
	else:
		print "No encoder present"
		return 0
	if os.path.isfile(os.getcwd()+"/Checkpoints/decoder"):
		decoder = torch.load(os.getcwd()+"/Checkpoints/decoder")
	else:
		print "No decoder present"
		return 0
	
	while 1:
		input_sentence = str(raw_input())
		encoder.initHidden()
		#Encoder Layer
		inp_encoder = embedding(input_sentence+ " " + const.EOS_Token)#.cuda()
		tmp, h = encoder(inp_encoder)
		# print inp_encoder

		#Decoder Layer
		#inp_decoder = const.SOS_Token + " " + target_sentence
		#tar_decoder = target_sentence + " " + const.EOS_Token
		inp_embed = embedding(const.SOS_Token)#.cuda()
		#tar_indexes = embedding.generateID(tar_decoder)#.cuda()
		#print inp_embed[1]
		decoder.hidden = h
		#decoder.initHidden()
		pword = ""
		out_str = ""
		while pword!=const.EOS_Token:
			out_str += pword + " "
			print pword
			out, h = decoder(inp_embed)
			# print out.clone()
			ind = out.data.topk(1)[1][0][0]
			pword = embedding.itoword(ind)
			inp_embed = embedding(pword)
		print out_str
	# return out_str


def main():
	train_mem("F1.txt",n_epochs=100)


if __name__ == "__main__":
	main()