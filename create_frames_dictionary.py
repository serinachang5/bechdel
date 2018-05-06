import csv
import pickle

from nltk.stem.porter import PorterStemmer

# iteratetes through agency/power verbs
# and creates dictionary to store each stemmed verb
# and its associated agency and power

def get_data(dataset):

	all_ = []
	for row in dataset:
		all_.append([row["verb"], row["agency"], row["power"]])
	return all_

def create_frames_dict(frames):

	stemmer = PorterStemmer()
	frames_dict = {}
	for f in frames:
		# stem verbs
		f[0] = stemmer.stem(f[0])
		frames_dict[f[0]] = list(f[1:])
	return frames_dict

if __name__ == "__main__":

	frames = csv.DictReader(open("data/agency_power.csv"))
	cleaned_frames = get_data(frames)

	res = create_frames_dict(cleaned_frames)

	# store frames dictionary in pickled file
	with open("frames.pickle", 'wb') as handle:
		pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
