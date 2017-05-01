import vocab
import framework as fw
import constants as const
from collections import Counter
import torch
import torch.nn as nn
import torch.autograd as ag
from torch.autograd import Variable

def train(input, target, encoder, decoder, encoder_opt, decoder_opt, crit, embedding, mod = None):
	encoder.init_hidden()
	encoder_opt.zero_grad()
	decoder_opt.zero_grad()

	loss = 0

	tmp, h = encoder(input)

	if mod == None:
		decoder.hidden = h
	else:
		pass

	inps = [SOS_Token] + target.split() + [EOS_Token]
	for i in range(len(inps)-1):
		out, h = decoder(inps[i])
		loss += crit(out[0].embedding.wordtoi[inps[i+1]])
	loss.backward()

	encoder_opt.step()
	decoder_opt.step()


def evaluate(input, target, encoder, decoder, embedding, mod=None):
	pass