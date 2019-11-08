import pickle
import io
import os
import csv
import numpy as np

import tokenization


def load_vectors(fname,vocab_size,embedding_size):
	fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
	n, d = map(int, fin.readline().split())
	token2id = {}
	id2token = {}
	data_ndarray = np.ndarray((vocab_size+2,embedding_size))

	token2id["[PAD]"] = 0
	id2token[0] = "[PAD]"
	data_ndarray[0,:] = np.zeros([embedding_size])

	token_id = 1
	for line in fin:
		if token_id > vocab_size:
			break
		tokens = line.rstrip().split(' ')
		token2id[tokens[0]] = token_id
		id2token[token_id] = tokens[0]
		data_ndarray[token_id,:] = list(map(float, tokens[1:]))
		token_id += 1

	mean = np.mean(data_ndarray[1:,:],axis=-1)
	mean_of_mean = np.mean(mean)
	std = np.std(data_ndarray[1:,:],axis=-1)
	mean_of_std = np.mean(std)

	token2id["[UNK]"] = token_id
	id2token[vocab_size] = "[UNK]"
	data_ndarray[token_id,:] = np.random.normal(mean_of_mean,mean_of_std,[embedding_size])
	return (token2id,id2token,data_ndarray)

def read_tsv(input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			print ("in csv file: ")
			print(len(lines))
			return lines

def get_all_data(data_dir,PairIndexDict):

	# for MNLI, training and developing files are named as matched and unmatched
	
	dev_file_name = ""
	test_file_name = ""

	if PairIndexDict[0] == 8:
		dev_file_name = "dev_matched.tsv"
		test_file_name = "test_matched.tsv"
	else:
		dev_file_name = "dev.tsv"
		test_file_name = "test.tsv"


	def collect_pairs(items,premises,hypothesis,PairIndexDict,max_len_p=0,max_len_h=0):
		for i,item in enumerate(items):
			if i>0:
				p = item[PairIndexDict[0]]
				h = item[PairIndexDict[1]]
				premises.append(p)
				hypothesis.append(h)
				max_len_p = np.maximum(max_len_p,len(p))
				max_len_h = np.maximum(max_len_h,len(h))
				

		return (premises,hypothesis,max_len_p,max_len_h)

	premises = []
	hypothesis = []

	train_pairs = read_tsv(os.path.join(data_dir, "train.tsv"))
	dev_pairs = read_tsv(os.path.join(data_dir, dev_file_name))
	test_pairs = read_tsv(os.path.join(data_dir, test_file_name))

	premises,hypothesis,max_len_p,max_len_h = collect_pairs(train_pairs, premises, hypothesis, PairIndexDict)
	print(len(premises))
	premises,hypothesis,max_len_p,max_len_h = collect_pairs(dev_pairs, premises, hypothesis, PairIndexDict, max_len_p, max_len_h)
	print(len(premises))
	premises,hypothesis,max_len_p,max_len_h = collect_pairs(test_pairs, premises, hypothesis, PairIndexDict, max_len_p, max_len_h)
	print(len(premises))

	premises.extend(hypothesis)
	print(len(premises))

	return premises, max_len_p, max_len_h

def build_char_vocabs(data_dir, char_embedding_table, char_dict, PairIndexDict):

	all_sentences,max_len_p,max_len_h = get_all_data(data_dir,PairIndexDict)
	print(len(all_sentences))
	tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
	_,char_embedding_size = char_embedding_table.shape

	chars2id = {}
	id2chars = {}
	tokenC_embeddings = []

	chars2id["[PAD]"] = 0
	id2chars[0] = "[PAD]"
	tokenC_embeddings.append(np.zeros([30]))
	token_id = 1

	for sen in all_sentences:
		
		tokens = tokenizer.tokenize(sen)
		chars_embedding = np.ndarray((len(tokens),char_embedding_size))
		for token in tokens:
			if token not in chars2id:
				chars2id[token] = token_id
				id2chars[token_id] = token
				token_id += 1

				chars = list(token)
				word_chars_embedding = np.ndarray((len(chars),char_embedding_size))
				for i,char in enumerate(chars):
					if char not in char_dict:
						word_chars_embedding[i,:] = char_embedding_table[-1,:]
					else:
						word_chars_embedding[i,:] = char_embedding_table[char_dict[char],:]
				word_char_embedding = np.amax(word_chars_embedding,axis=0)
				tokenC_embeddings.append(word_char_embedding)

	token_char_embedding_table = np.ndarray((len(tokenC_embeddings),char_embedding_size))
	for i in range(len(tokenC_embeddings)):
		token_char_embedding_table[i,:] = tokenC_embeddings[i]

	return (chars2id,id2chars,token_char_embedding_table)

if __name__ == "__main__":

	data_dir = ["MNLI","SNLI"]
	PairIndexDict = {"MNLI":(8,9),"SNLI":(7,8)}
	char_dict = {}
	for c in range(26):
		char_dict[str(chr(c+97))] = c
	for n in range(26,26+10):
		char_dict[str(chr(n+48))] = n

	char_embedding_table = np.random.uniform(-0.25,0.25,(37,30))
	for task in data_dir:
		(chars2id,id2chars,token_char_embedding_table) = build_char_vocabs(task,char_embedding_table,char_dict,PairIndexDict[task])
		with open("vocab_" + task + "_file.txt","w") as f:
			for token in chars2id:
				f.write(token + "\n")

		pickle.dump([chars2id,id2chars,token_char_embedding_table],open(task + "-char_vocab_embedding.p","wb"))
		print("task: " + task)
		print(token_char_embedding_table.shape[0])

	
	(token2id,id2token,embedding_table) = load_vectors("wiki-news-300d-1M.vec",30000,300)
	pickle.dump([token2id,id2token,embedding_table],open("wiki-news-300d-30k.p","wb"))

	with open("vocab_file.txt","w") as f:
		for token in token2id:
			f.write(token + "\n")
	
