import argparse
import pickle
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn
from collections import Counter

## download WordNet 2.0
# https://wordnetcode.princeton.edu/2.0/
# https://stackoverflow.com/questions/29332454/python-3-and-nltk-with-wordnet-2-1-is-that-possible
wn2 = WordNetCorpusReader("../WordNet-2.0/dict", "../WordNet-2.0/dict")

def get_data():
	print("loading data from dictionary mapping...")

	# save filename
	domains = pickle.load( open("data/processed/domain-mapping.p", "rb" ) )

	return domains

# https://stackoverflow.com/questions/13881425/get-wordnets-domain-name-for-the-specified-word
def get_offsets(words):
	offsets = {}
	for w in words:
		# print("WORD: " + str(w))
		syn = wn2.synsets(w)
		# print(syn)
		if len(syn) != 0:
		# print("ANCESTOR: " + str(syn.hypernyms()))
			offset = wn2.synsets(w)[0].offset()
		else:
			offset = "n/a"
		# offset = syn.hypernyms()[0].offset()
		offsets[w] = offset
	print(" # of offsets: " + str(len(offsets)))
	return offsets 

def get_categories(words, offsets, domains):
	print("loading...")

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

def get_categories_list(words, offsets, domains):
	print("loading...")

	categories = []
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
		categories.append(category)
	return categories, flatten_categories

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', dest='data', type=str, help='path to data', required=True)
	args = parser.parse_args()

	print(wn2.get_version())

	file_loc = "data/processed/" + str(args.data)
	file_name = "data/processed/" + args.data.split("/")[-1].split(".")[0] + "_categories"
	data = pickle.load( open( file_loc, "rb" ) )

	domains = get_data()
	print(" # of words: " + str(len(domains.keys())))
	all_categories = []
	for c in domains.values():
		all_categories.extend(c)
	print(" # of wordnet categories: " + str(len(list(set(all_categories)))))

	offsets = get_offsets(data)
	categories, flatten_categories = get_categories(data, offsets, domains)
	print(" # of unique categories in given words: " + str(len(list(flatten_categories))))
	pickle.dump( categories, open(file_name + ".p", "wb" ) )
	# print(categories)
	# print(Counter(categories).keys())
	# 
	print("done.")


if __name__ == '__main__':
	main()