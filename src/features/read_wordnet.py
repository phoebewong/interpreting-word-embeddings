import pickle

# download first: http://wndomains.fbk.eu/index.html
# 200 categories: http://wndomains.fbk.eu/hierarchy.html

fh = open('../wn-domains-3.2/wn-domains-3.2-20070223', 'r')
dbdomains = {}
count = 0
for line in fh:
    offset, domain = line.split('\t')
    dbdomains[offset[:-2]] = domain[:-1]
    count+=1
fh.close()
print("# of words: " + str(count))

pickle.dump( dbdomains, open("data/processed/domain-mapping.p", "wb" ) )

print("done.")

# other wordnet ontology: https://medium.com/@pragadesh/semantic-similarity-using-wordnet-ontology-b12219943f23
# similarities: https://stackoverflow.com/questions/41793842/wordnet-python-words-similarity/41794714
# topic domains: http://www.nltk.org/howto/wordnet.html
# genism: https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity