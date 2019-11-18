import argparse
import pickle
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

def get_data(filename):
	print("loading data from " + filename )

	# save filename
	words = pickle.load( open( filename, "rb" ) )

	print(" # of words: " + str(len(words)))
	return words

def get_features(words):
	print("loading features...")

	all_features = []

	for token in words:
		word = nlp(token)

		# labels
		features = [word[0].pos_, word[0].tag_, word[0].dep_, word[0].is_alpha, word[0].is_stop]

		# named entities
		if len(word.ents) == 0:
			features.append("None")
		else:
			features.append(word.ents[0].label_)

		# sentiments
    	vader_analyzer = SentimentIntensityAnalyzer()
    	scores = vader_analyzer.polarity_scores(token)
    	features.extend([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

		all_features.append(features)
	return np.array(all_features)

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', dest='data', type=str, help='path to data', required=True)
	args = parser.parse_args()

	file_loc = "data/processed/" + str(args.data)

	words = get_data(file_loc)

	# get spacy features
	nlp = spacy.load("en_core_web_sm")
	
	print("done.")


if __name__ == '__main__':
	main()