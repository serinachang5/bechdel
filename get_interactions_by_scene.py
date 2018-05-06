import csv
import os
import pickle

def get_sample(dataset):
	sample = []
	count = 0
	for row in dataset:
		if row["IMDb_id"] != "":
			count += 1
			sample.append((row["Title"], row["Source"], row["File_path"], row["IMDb_id"], \
				row["Bechdel_rating"], row["Year"]))
		if count >= 2:
			break
	return sample

def get_all(dataset):
	all_ = []
	for row in dataset:
		if row["IMDb_id"] != "":
			all_.append((row["Title"], row["Source"], row["File_path"], row["IMDb_id"], \
				row["Bechdel_rating"], row["Year"]))
	return all_

# get interactions for agarwal files
def get_interactions_agarwal(agarwal, agarwal_gender_dict):

	movies_with_scenes = []
	movies_without_scenes = []

	mm_interactions = []
	fm_interactions = []
	ff_interactions = []

	movie_scenes_dict = {}
	match_gender_to_scenes = []

	for item in agarwal:
		if os.path.exists(item[2][:-4]+item[0]+"_scenes"):
			movies_with_scenes.append(item[2][:-4]+item[0]+"_scenes")
		else:
			movies_without_scenes.append(item[2][:-4]+item[0]+"_scenes")

	for m in movies_with_scenes:
		dirs = os.listdir(m)
		movie_scenes_dict[m] = dirs

	for key, val in movie_scenes_dict.items():
		sp = key.split("/")
		for k in list(agarwal_gender_dict.keys()):
			if sp[4][:24] in k:
				match_gender_to_scenes.append((k,key))

	count_chars = 0
	seen = set()
	fin = []
	for matches in match_gender_to_scenes:
		dirs = os.listdir(matches[1])
		for d in dirs:
			chars = agarwal_gender_dict[matches[0]]
			with open(matches[1]+"/"+d) as fp:
				contents = fp.readlines()

				print(matches[1]+"/"+d,chars, contents)
	# 			for c in chars:
	# 				gender = c[0]
	# 				variants = c[1:]
	# 				for v in variants:
	# 					if any(v in c for c in contents):
	# 						count_chars += 1
	# 						if count_chars == 2:
	# 							fin.append(matches[1]+"/"+d)
	# return fin


if __name__ == "__main__":

	agarwal = csv.DictReader(open("data/agarwal_alignments_with_IDs_with_bechdel.csv"))
	gorinski = csv.DictReader(open("data/gorinski_alignments_with_IDs_with_bechdel.csv"))

	with open('agarwal_chars_by_gender.pickle', 'rb') as handle:
		a_char_gen = pickle.load(handle)

	# all_movies = getAll(agarwal)
	all_movies = get_sample(agarwal)
	m = get_interactions_agarwal(all_movies, a_char_gen)
	print(m)






