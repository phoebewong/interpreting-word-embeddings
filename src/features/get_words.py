import argparse
import pickle

def get_data(filename, output_path):
	print("loading data from " + filename )

	# reading in data from string format
	lines = open(filename).readlines()
	words = []
	for line in lines:
		tokens = line.strip().split()
		words.append(tokens[0])

	# print data specs
	print("loaded data.")
	print(" #words = %d " %(len(words)) )

	# save file
	pickle.dump( words, open(output_path + ".p", "wb" ) )
	return

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', dest='data', type=str, help='path to data', required=True)
	args = parser.parse_args()

	file_loc = "data/external/" + str(args.data)
	output_path = "data/processed/" + args.data.split("/")[-1].split(".")[0]
	print("saved file: " + str(output_path))

	get_data(file_loc, output_path)
	print("done.")


if __name__ == '__main__':
	main()