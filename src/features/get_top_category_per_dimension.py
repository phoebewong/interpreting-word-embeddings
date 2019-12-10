import gensim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import argparse
import pickle
from nltk.corpus import WordNetCorpusReader
from collections import Counter
import seaborn as sns
import os
import argparse

wn2 = WordNetCorpusReader("../WordNet-2.0/dict", "../WordNet-2.0/dict")

if not os.path.exists('../../../images/'):
	os.makedirs('../../../images/')

def get_top_words(embed_list, words, n):
	top_indices = embed_list.argsort()[-n:][::-1]
	top_words = [words[i] for i in top_indices]
	return top_words

def evaluate_dimensions(embed_matrix, words, n):
	dimension_count = {}
	print("shape: " + str(np.shape(embed_matrix)))
	for d in range(0, np.shape(embed_matrix)[1]):
		col = embed_matrix[:,d]
		top_words = get_top_words(col, words, n)
		dimension_count[d] = top_words
	return dimension_count

def get_data():
	print("loading data from dictionary mapping...")
	domains = pickle.load( open("data/processed/domain-mapping.p", "rb" ) )
	return domains

# https://stackoverflow.com/questions/13881425/get-wordnets-domain-name-for-the-specified-word
def get_offsets(words):
	offsets = {}
	for w in words:
		syn = wn2.synsets(w)
		if len(syn) != 0:
			offset = wn2.synsets(w)[0].offset()
		else:
			print("missing word: " + str(w))
			offset = "n/a"
		offsets[w] = offset
#     print(" # of offsets: " + str(len(offsets)))
	return offsets 

def get_categories(words, offsets, domains):
	categories = {}
	flatten_categories = set()
	for w in words:
		o = offsets[w]
		k = str(o).zfill(8)
		if k in domains:
			category = domains[k]
			if len(category) != 1:
				for c in category:
					flatten_categories.add(c)
		else:
			category = "n/a"
		categories[w] = category
	return categories, flatten_categories

def get_all_values(d):
	if isinstance(d, dict):
		for v in d.values():
			yield from get_all_values(v)
	elif isinstance(d, list):
		for v in d:
			yield from get_all_values(v)
	else:
		yield d 

def get_domain_count(domains, dimension_dict, spine_embeddings, spine_tokens):
	dimension_to_category_map = {}
	for i in tqdm(range(len(dimension_dict.keys()))):
		caetgory_counter = {}
		data = dimension_dict[i]
		offsets = get_offsets(data)
		categories, _ = get_categories(data, offsets, domains) 
		vals = list(get_all_values(categories))
		dimension_to_category_map[i] = Counter(vals)
	return dimension_to_category_map

def convert_to_df(col, category_labels):
	missing_categories = set(category_labels).difference(set(col.keys()))
	missing_dict = {}
	for mc in missing_categories:
		missing_dict[mc] = 0
	col.update(missing_dict)
	df_col = pd.DataFrame(list(col.items()), columns=['domain', 'count'])
	return df_col

def plot_graphs(dimension_to_category_map, category_labels, embedding_type=""):
	num_dimensions = len(dimension_to_category_map)
	for i in tqdm(range(num_dimensions)):
		plt.clf()
		plt.rcParams["xtick.labelsize"] = 3
		col = dimension_to_category_map[i]
		df_col = convert_to_df(col, category_labels)
		graph = sns.barplot(x='domain', y="count", data=df_col, order=sorted(df_col['domain']))
		graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
		plt.title("dimension " + str(i))
		plt.tight_layout()
		plt.savefig("../../../images/dimension-" + str(i) + "-" + embedding_type + ".png")
	return

def get_embeddings(file, n):
	spine = open("data/external/" + str(file),"r") .read().split('\n')
	spine.pop(15000) # remove the last empty object
	print(len(spine))

	spine_tokens = []
	spine_embeddings = []

	for i, line in enumerate(spine):
		tokens = line.strip().split()
		spine_tokens.append(tokens[0])
		spine_embeddings.append([float(i) for i in tokens[1:]])

	spine_tokens = np.array(spine_tokens)
	spine_embeddings = np.array(spine_embeddings)

	dimension_dict = evaluate_dimensions(spine_embeddings, spine_tokens, n)
	return dimension_dict, spine_tokens, spine_embeddings

def specific_dimension(i, dimension_to_category_map, category_labels):
	num_dimensions = len(dimension_to_category_map)
	col = dimension_to_category_map[i]
	df_col = convert_to_df(col, category_labels)
	return df_col

def find_top_domains(indices, word, dimension_to_category_map):
	print("WORD: " + str(word))
	for i in indices:
		print("INDEX: " + str(i))
		specific_col = specific_dimension(i, dimension_to_category_map, category_labels)
		specific_col = specific_col.sort_values(by='count', ascending=False)
		# print(" total domains: " + specific_col_wv['count'])
		print(specific_col.head())
		print()
	return

def save_to_pickle(val, file_name):
	pickle.dump( val, open(file_name + ".p", "wb" ) )
	return

def find_top_domain_of_all(dict_map):
	top_map = {}
	for key in dict_map:
		val_list = dict_map[key].most_common(1)[0]
		top_map[key] = val_list[0]
	return top_map

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-top",  "--top", action='store_true', default=False, help="True if find top, False if not")
	parser.add_argument("-use_saved",  "--use_saved", action='store_true', default=False, help="True if recalculate, False if not")
	args = parser.parse_args()

	nums = [1,10,50,100]

	if args.use_saved and args.top:
		print("using saved and find top...")
		for n in nums:
			print("NUMBER: " + str(n))
			print("glove ")
			glove_dimension_to_category_map = pickle.load(open("data/raw/all_top_" + str(n) + "_category_glove.p", "rb"))
			top_map = find_top_domain_of_all(glove_dimension_to_category_map)
			save_to_pickle(top_map, "data/raw/top_" + str(n) + "_category_glove")

			print("word2vec ")
			wv_dimension_to_category_map = pickle.load(open("data/raw/all_top_" + str(n) + "_category_word2vec.p", "rb"))
			top_map = find_top_domain_of_all(wv_dimension_to_category_map)
			save_to_pickle(top_map, "data/raw/top_" + str(n) + "_category_word2vec")

	else:
		# DOMAINS
		domains = get_data()
		print(" # of words: " + str(len(domains.keys())))
		all_categories = []
		for c in domains.values():
			all_categories.extend(c)
		print(" # of wordnet categories: " + str(len(list(set(all_categories)))))

		category_labels = sorted(list(set(all_categories)))
		print(category_labels[:10])

		for n in nums:
			print("NUMBER: " + str(n))
			# GLOVE
			print("GLOVE")
			glove_dimension_dict, glove_spine_embeddings, glove_spine_tokens = get_embeddings("SPINE_glove.txt", n)
			glove_dimension_to_category_map = get_domain_count(domains, glove_dimension_dict, glove_spine_embeddings, glove_spine_tokens)
			save_to_pickle(glove_dimension_to_category_map, "data/raw/all_top_" + str(n) + "_category_glove")

			# WORD2VEC
			print("WORD2VEC")
			wv_dimension_dict, wv_spine_embeddings, wv_spine_tokens = get_embeddings("SPINE_word2vec.txt", n)
			wv_dimension_to_category_map = get_domain_count(domains, wv_dimension_dict, wv_spine_embeddings, wv_spine_tokens)
			save_to_pickle(wv_dimension_to_category_map, "data/raw/all_top_" + str(n) + "_category_word2vec")

	print("done.")
	return

if __name__ == "__main__":
	main()