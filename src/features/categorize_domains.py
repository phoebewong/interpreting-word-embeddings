import argparse
import pickle
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

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
	offsets = []
	for w in words:
		# print("WORD: " + str(w))
		syn = wn2.synsets(w)[0]
		# print("ANCESTOR: " + str(syn.hypernyms()))
		offset = wn2.synsets(w)[0].offset()
		# offset = syn.hypernyms()[0].offset()
		offsets.append(offset)
	return offsets 

def get_categories(offsets, domains):
	print("loading...")

	categories = []
	for o in offsets:
		k = str(o).zfill(8)
		if k in domains:
			category = domains[k]
		else:
			category = "n/a"
		categories.append(category)
	return categories

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', dest='data', type=str, help='path to data', required=True)
	args = parser.parse_args()

	print(wn2.get_version())

	# file_loc = "data/processed/" + str(args.data)
	# data = pickle.load( open( file_loc, "rb" ) )
	data = ['bag', 'luggage', 'purse', 'bags', 'compartment', 'envelope', 'pocket', 'belongings', 'closet', 'cartridge']
	# data = ['entrepreneurs', 'angel', 'ventures', 'collaborations', 'innovation', 'grants', 'talent', 'partnerships', 'incentives', 'opportunities']
	data = ['art', 'religion', 'mathematics', 'internet']
	# data = ['travel_guidebook', 'programmer']

	domains = get_data()
	print(" # of words: " + str(len(domains.keys())))
	print(list(domains.keys())[:5])
	# print(domains['00506605'])
	offsets = get_offsets(data)
	categories = get_categories(offsets, domains)
	print(categories)
	
	print("done.")


if __name__ == '__main__':
	main()