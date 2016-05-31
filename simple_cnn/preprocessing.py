#!/usr/bin/python
# -*- coding:utf8 -*-

# Change discourse output.
# Save (any) w2v as output.

import json, re
import h5py
import numpy as np

label2idx = {"Instantiation":0,"Synchrony":1,"Pragmatic cause":2,"List":3,"Asynchronous":4,"Restatement":5,"Concession":6,"Conjunction":7,"Cause":8,"Alternative":9,"Contrast":10}

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)     
	string = re.sub(r"\'s", " is", string) 
	string = re.sub(r"\'ve", " have", string) 
	string = re.sub(r"n\'t", " not", string) 
	string = re.sub(r"\'re", " are", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " will", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " ( ", string) 
	string = re.sub(r"\)", " ) ", string) 
	string = re.sub(r"\?", " ? ", string) 
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip().lower()

def get_labels(label_seg):
	labels = []
	for label in label_seg:
		labels.add(label.split(".")[-1])
	return labels

def get_words(jsn):
	tmp = clean_str(jsn["RawText"]).split()
	words = []
	for word in tmp:
		if u"\u00a0" in word:
			word = "NULL"
		words.append(word)
	return words

def get_discourse(jsn, r, maxlen, train=False):
	"""
	return a list containint a pair where the first
	element is a pair of two subsentences and the 
	second element is a vector stands for their 
	label (one hot)
	"""
	words1 = get_words(jsn["Arg1"])
	words2 = get_words(jsn["Arg2"])
	idx1 = []
	idx2 = []

	for word in words1:
		idx1.append(r.word2idx(word))
	if len(idx1) > maxlen:
		print len(idx1)
		idx1 = idx1[:maxlen]
	pad_num = maxlen - len(idx1)
	idx1 = [0] * ((pad_num+1)/2) + idx1 + [0] * (pad_num/2)
	for word in words2:
		idx2.append(r.word2idx(word))
	if len(idx2) > maxlen:
		print len(idx2)
		idx2 = idx2[:maxlen]
	pad_num = maxlen - len(idx2)
	idx2 = [0] * ((pad_num+1)/2) + idx2 + [0] * (pad_num/2)

	i = [idx1, idx2]

	labels = []
	for label in jsn["Sense"]:
		labels.append(label2idx[label.split(".")[-1]])
	if train != True:
		o = [0]*12
		for label_idx in labels:
			o[label_idx] = 1
		io = [i, o]
		return io
	else:
		io_list = []
		for label_idx in labels:
			o = [0] * 12
			o[label_idx] = 1
			io = [i, o]
			io_list.append(io)
		return io_list


class Vocab(object):
	"""Build the vocab of inputs"""
	def __init__(self, conf):
		self.file = conf.get("vocab_file", "")
		if self.file == "":
			self.file = conf["train_file"]
		self.idx = ["NULL"]
		self.word = {0: "NULL"}
		self.vocab_size = 1
		self.vec_size = conf["vec_size"]
		self.gconf = conf
		self.ml = conf["maxlen"]

	def word2idx(self, word):
		return self.word.get(word, 0)

	def idx2word(self, idx):
		if idx >= len(self.idx):
			return self.word[0]
		return self.word[idx]

	def add_word(self, word):
		if self.word2idx(word) == 0:
			self.word[word] = len(self.idx)
			self.idx.append(word)
			self.vocab_size += 1

	def read_vocab(self):
		f = open(self.file, "r")
		if self.file == self.gconf["train_file"]:
			line = f.readline()
			while line != "":
				jsn = json.loads(line)
				words = get_words(jsn["Arg1"])
				words += get_words(jsn["Arg2"])
				for word in words:
					self.add_word(word)
				line = f.readline()
		else:
			line = f.readline()
			while line != "":
				self.add_word(line[:-1])
				line = f.readline()
		f.close()
		print "Vocab read..."

	def save_vocab(self):
		f = open("data/vocab.save", "w")
		for word in self.idx[1:]:
			try:
				f.write(word+"\n")
			except:
				print word
		f.close()
		print "Vocab saved"

	def save_w2v(self):
		f = hdf5.File("data/w2v.hdf5", "w")
		w2v = np.random.rand(self.vocab_size, self.vec_size) - 0.5
		f["w2v"] = w2v
		f.close()

class Reader(object):
	"""This is a reader"""
	def __init__(self, conf):
		self.train_file = conf["train_file"]
		# self.test_file = conf["test_file"]
		self.valid_file = conf["dev_file"]
		self.ml = conf['maxlen']
		self.train = {"arg1": [], "arg2": [], "label": []}
		self.dev = {"arg1": [], "arg2": [], "label": []}
		self.vocab = Vocab(conf)
		self.vocab.read_vocab()

	def get_full_train_data(self):
		f = open(self.train_file, "r")
		line = f.readline()
		while line != "":
			jsn = json.loads(line)
			if jsn["Type"] != "Implicit":
				line = f.readline()
				continue
			gd = get_discourse(jsn, self.vocab, self.ml, train=True)
			for item in gd:
				self.train["arg1"].append(item[0][0])
				self.train["arg2"].append(item[0][1])
				self.train["label"].append(item[1])
			line = f.readline()
		print "training data ready..."
		return self.train

	def get_full_valid_data(self):
		f = open(self.valid_file, "r")
		line = f.readline()
		while line != "":
			jsn = json.loads(line)
			if jsn["Type"] != "Implicit":
				line = f.readline()
				continue
			item = get_discourse(jsn, self.vocab, self.ml)
			self.dev["arg1"].append(item[0][0])
			self.dev["arg2"].append(item[0][1])
			self.dev["label"].append(item[1])
			line = f.readline()
		print "validation data ready..."
		return self.dev

	def dump_train(self):
		print 1
		self.get_full_train_data()
		print 2

		f = h5py.File("data/pdtv_train.hdf5", "w")

		f["arg1"] = np.array(self.train["arg1"])
		f["arg2"] = np.array(self.train["arg2"])
		f["label"] = np.array(self.train["label"])

		f.close()

	def dump_valid(self):
		self.get_full_valid_data()

		f = h5py.File("data/pdtv_dev.hdf5", "w")

		f["arg1"] = np.array(self.dev["arg1"])
		f["arg2"] = np.array(self.dev["arg2"])
		f["label"] = np.array(self.dev["label"])

		f.close()

	def dump_w2v(self):
		self.vocab.save_w2v()

if __name__ == '__main__':
	maxlen = 200

	conf = {
		"train_file": "../data/train_pdtb.json",
		"dev_file": "../data/dev_pdtb.json",
		"w2v_file": "../data/glove.bin",
		# "vocab_file": "data/vocab",
		"test_file": "",
		# "vocab_size": 100000,
		"vec_size": 100,
		"maxlen": maxlen
	}
	# test vocab
	# r = Vocab(conf)
	# r.read_vocab()
	# print r.vocab_size

	# # test discourse reader
	# jsn = json.loads("""{"DocID": "wsj_2201", "Arg1": {"RawText": "It considered running them during tomorrow night's World Series broadcast but decided not to when the market recovered yesterday", "NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "Word": ["It", "considered", "running", "them", "during", "tomorrow", "night", "'s", "World", "Series", "broadcast", "but", "decided", "not", "to", "when", "the", "market", "recovered", "yesterday"], "POS": ["PRP", "VBD", "VBG", "PRP", "IN", "NN", "NN", "POS", "NNP", "NNP", "NN", "CC", "VBD", "RB", "TO", "WRB", "DT", "NN", "VBD", "NN"], "Lemma": ["it", "consider", "run", "they", "during", "tomorrow", "night", "'s", "World", "Series", "broadcast", "but", "decide", "not", "to", "when", "the", "market", "recover", "yesterday"]}, "Arg2": {"RawText": "Other brokerage firms, including Merrill Lynch & Co., were plotting out potential new ad strategies", "NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "Word": ["Other", "brokerage", "firms", ",", "including", "Merrill", "Lynch", "&", "Co.", ",", "were", "plotting", "out", "potential", "new", "ad", "strategies"], "POS": ["JJ", "NN", "NNS", ",", "VBG", "NNP", "NNP", "CC", "NNP", ",", "VBD", "VBG", "RP", "JJ", "JJ", "NN", "NNS"], "Lemma": ["other", "brokerage", "firm", ",", "include", "Merrill", "Lynch", "&", "Co.", ",", "be", "plot", "out", "potential", "new", "ad", "strategy"]}, "Connective": {"RawText": ["meanwhile"]}, "Sense": ["Expansion.Conjunction", "Temporal.Synchrony"], "Type": "Implicit", "ID": "35975"}
	#      """)
	# print get_discourse(jsn, r, maxlen)

	# test reader
	reader = Reader(conf)
	reader.dump_train()
	reader.dump_valid()
