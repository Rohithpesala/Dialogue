import vocab
import framework as fw
import constants as const
from collections import Counter
import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable

def train_lstm(input_sentence, target_sentence, encoder, decoder, encoder_opt, decoder_opt, crit, embedding):
	"""
	Parameters:
	::
	Return:
	:: Loss
	"""

	encoder.initHidden()
	encoder_opt.zero_grad()
	decoder_opt.zero_grad()

	loss = 0

	tmp, h = encoder(input_sentence)

	decoder.hidden = h
	
	inps = [const.SOS_Token] + target_sentence.split() + [const.EOS_Token]
	for i in range(len(inps)-1):
		out, h = decoder(inps[i])

		#print out[0],embedding.vcb.stoi[inps[i+1]],inps[i+1]
		print out.clone(),embedding.vcb.stoi[inps[i+1]]
		loss += crit(out[0],ag.Variable(torch.LongTensor([embedding.vcb.stoi[inps[i+1]]])))
		#return 0
	#print "s"
	#return 0
	loss.backward()
	#print "a"
	encoder_opt.step()
	#print "b"
	decoder_opt.step()
	#print "c"
	return loss


def evaluate(input, target, encoder, decoder, embedding, mod=None):
	pass