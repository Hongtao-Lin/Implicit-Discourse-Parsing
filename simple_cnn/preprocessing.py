#!/usr/bin/python
# -*- coding:utf8 -*-

# Change discourse output.
# Save (any) w2v as output.

import json, re
import h5py
import numpy as np

label2idx = {"Instantiation":0,"Synchrony":1,"Pragmatic cause":2,"List":3,"Asynchronous":4,"Restatement":5,"Concession":6,"Conjunction":7,"Cause":8,"Alternative":9,"Contrast":10}
# label2idx = {"Instantiation":0,"Synchrony":4,"Pragmatic cause":4,"List":4,"Asynchronous":4,"Restatement":1,"Concession":4,"Conjunction":2,"Cause":3,"Alternative":4,"Contrast":4}

POS_list = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", \
		    "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH" \
		    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WDN", "WP", "WP$", "WRB"]
POS2idx = {}

i = 1
for p in POS_list:
	POS2idx[p] = i
	i += 1

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
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
	words = []
	for i in range(len(jsn["Word"])):
		words.append(jsn["Word"][i])
		words.append("POS_" + jsn["POS"][i])
	# words = jsn["Word"]
	# for pos in jsn["POS"]:
	# 	words.append("POS_"+pos)
	return words

def get_POS(jsn, start):
	pos = []
	for p in jsn["POS"]:
		if p not in POS_list:
			pos.append(0+start)
		else:
			pos.append(POS2idx[p]+start)
	return pos

def get_discourse(jsn, r, maxlen, train=False, POS_concat=False):
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
	start = 1
	if POS_concat:
		pos1 = get_POS(jsn["Arg1"], start)
		pos2 = get_POS(jsn["Arg2"], start)

	for word in words1:
		idx1.append(r.word2idx(word)+start)
	if len(idx1) > maxlen:
		print "arg1 length: ", len(idx1)
		idx1 = idx1[:maxlen]
	pad_num = maxlen - len(idx1)
	idx1 = [0+start] * ((pad_num+1)/2) + idx1 + [0+start] * (pad_num/2)
	for word in words2:
		idx2.append(r.word2idx(word)+start)
	if len(idx2) > maxlen:
		print "arg2 length: ", len(idx2)
		idx2 = idx2[:maxlen]
	pad_num = maxlen - len(idx2)
	idx2 = [0+start] * ((pad_num+1)/2) + idx2 + [0+start] * (pad_num/2)

	if POS_concat:
		if len(pos1) > maxlen:
			pos1 = pos1[:maxlen]
		pad_num = maxlen - len(pos1)
		pos1 = [0+start] * ((pad_num+1)/2) + pos1 + [0+start] * (pad_num/2)
		if len(pos2) > maxlen:
			pos2 = pos2[:maxlen]
		pad_num = maxlen - len(pos2)
		pos2 = [0+start] * ((pad_num+1)/2) + pos2 + [0+start] * (pad_num/2)


	i = [idx1, idx2]

	labels = []
	num_classes = max(label2idx.values())+1
	for label in jsn["Sense"]:
		labels.append(label2idx[label.split(".")[-1]])
	if train != True:
		o = [0+start]*num_classes
		for label_idx in labels:
			o[label_idx] += 1
		io = [i, o]
		if POS_concat:
			io.append([pos1, pos2])
		return io
	else:
		io_list = []
		for label_idx in labels:
			o = [0+start] * num_classes
			o[label_idx] += 1
			io = [i, o]
			if POS_concat:
				io.append([pos1, pos2])
			io_list.append(io)
		return io_list

class Vocab(object):
	"""Build the vocab of inputs"""
	def __init__(self, conf):
		self.file = conf.get("vocab_file", "")
		if self.file == "":
			self.file = conf["train_file"]
		self.idx = ["NULL"]
		self.word = {"NULL": 0} # {"a": 1}
		self.vocab_size = 1
		self.vec_size = conf["vec_size"]
		self.gconf = conf
		self.w2v = {}
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

	def load_w2v(self):
		fname = self.gconf["w2v_file"]
		if fname == "":
			return
		f = open(fname, "r")
		w2v = {}
		for line in f.readlines():
			word = line.split(" ")[0]
			vec = [float(v) for v in line.split(" ")[1:]]
			w2v[word] = vec
		print 1
		words = set.intersection(set(self.idx), set(w2v.keys()))
		print 2
		for word in self.idx:
			self.w2v[word] = [0]*self.vec_size
		for word in words:
			self.w2v[word] = w2v[word]
		print len(self.w2v), len(self.idx), len(w2v)
			# if word not in self.idx:
				# continue
		print "loading w2v complete!"

	def save_w2v(self):
		f = h5py.File("data/w2v.hdf5", "w")
		w2v = np.zeros([self.vocab_size, self.vec_size])
		if self.w2v == {}:
			w2v = np.random.rand(self.vocab_size, self.vec_size) - 0.5
		else:
			i = 0
			for w in self.idx:
				w2v[i] = self.w2v[w]
				i += 1
		f["w2v"] = w2v
		f.close()

class Reader(object):
	"""This is a reader"""
	def __init__(self, conf):
		self.train_file = conf["train_file"]
		# self.test_file = conf["test_file"]
		self.valid_file = conf["dev_file"]
		self.ml = conf['maxlen']
		self.train = {"arg1": [], "arg2": [], "label": [], "pos1": [], "pos2": []}
		self.dev = {"arg1": [], "arg2": [], "label": [], "pos1": [], "pos2": []}
		self.vocab = Vocab(conf)
		self.gconf = conf
		self.vocab.read_vocab()
		self.vocab.load_w2v()

	def get_full_train_data(self):
		f = open(self.train_file, "r")
		line = f.readline()
		while line != "":
			jsn = json.loads(line)
			if jsn["Type"] != "Implicit":
				line = f.readline()
				continue
			gd = get_discourse(jsn, self.vocab, self.ml, train=True, POS_concat=self.gconf["POS_concat"])
			for item in gd:
				self.train["arg1"].append(item[0][0])
				self.train["arg2"].append(item[0][1])
				if self.gconf["POS_concat"]:
					self.train["pos1"].append(item[2][0])
					self.train["pos2"].append(item[2][1])
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
			item = get_discourse(jsn, self.vocab, self.ml, POS_concat=self.gconf["POS_concat"])
			self.dev["arg1"].append(item[0][0])
			self.dev["arg2"].append(item[0][1])
			self.dev["label"].append(item[1])
			if self.gconf["POS_concat"]:
					self.dev["pos1"].append(item[2][0])
					self.dev["pos2"].append(item[2][1])
			line = f.readline()
		print "validation data ready..."
		return self.dev

	def dump_train(self):
		self.get_full_train_data()

		f = h5py.File("data/pdtb_train.hdf5", "w")

		f["arg1"] = np.array(self.train["arg1"])
		f["arg2"] = np.array(self.train["arg2"])
		f["label"] = np.array(self.train["label"])
		if self.gconf["POS_concat"]:
			f["pos1"] = np.array(self.train["pos1"])
			print f["pos1"]
			f["pos2"] = np.array(self.train["pos2"])

		f.close()

	def dump_valid(self):
		self.get_full_valid_data()

		f = h5py.File("data/pdtb_dev.hdf5", "w")

		f["arg1"] = np.array(self.dev["arg1"])
		f["arg2"] = np.array(self.dev["arg2"])
		f["label"] = np.array(self.dev["label"])
		if self.gconf["POS_concat"]:
			f["pos1"] = np.array(self.dev["pos1"])
			f["pos2"] = np.array(self.dev["pos2"])

		f.close()

	def dump_w2v(self):
		self.vocab.save_w2v()

if __name__ == '__main__':
	maxlen = 300

	conf = {
		"train_file": "../data/train_pdtb.json",
		"dev_file": "../data/dev_pdtb.json",
		# "w2v_file": "../data/glove.bin",
		"w2v_file": "",
		# "vocab_file": "data/vocab",
		"test_file": "",
		# "vocab_size": 100000,
		"vec_size": 150,
		# "POS_concat": True,
		"POS_concat": False,
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
	reader.dump_w2v()
	reader.dump_train()
	reader.dump_valid()
