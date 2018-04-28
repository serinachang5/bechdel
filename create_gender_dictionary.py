import os
import pickle

# iteratetes through movie_by_gender directory
# and creates dictionary to store each character
# name, its variants, and gender roles

# params: folder path
# returns: dictionary of movie and character info
def get_gender_by_movie(path):

	all_files = os.listdir(path)

	gender_in_movies = {}

	for filename in all_files:

		chars = []
		char_info = []
		start_end_char = []
		char_boundaries = []
		store_gender = []
		store_variants = []
		concat = []

		with open(path+filename) as fp:
			contents = fp.readlines()
			gender_in_movies_agarwal[contents[4][11:]] = []
			for idx, line in enumerate(contents):
				if line[0:5] == "Root:":
					chars.append(idx)
				char_info.append(line)

		for c in range(len(chars) - 1):
			current_item, next_item = chars[c], chars[c+1]
			start_end_char.append((current_item, next_item))

		for bound in start_end_char:
			char_boundaries.append(char_info[bound[0]:bound[1]])

		for idx, char in enumerate(char_boundaries):
			for i, c in enumerate(char):
				if "Gender" in c:
					store_gender.append(c[11:].strip())

		for idx, char in enumerate(char_boundaries):
			for i, c in enumerate(char):
				if "Variants:" in c:
					if len(char[i:]) != 1:
					 	store_variants.append(tuple(char[i+1:]))
		
		if len(store_variants) > 0 and len(store_gender) > 0:
			for g, v in zip(store_gender, store_variants):
				concat.append((g,) + v)
				new_concat = [':'.join(x).replace(' ', '').replace('\n', '') for x in concat]
			
				gender_in_movies[contents[4][11:]] = new_concat

	return gender_in_movies

if __name__ == "__main__":

	agarwal_chars_by_gender = get_gender_by_movie("data/movie_by_gender/agarwal/")
	gorinski_chars_by_gender = get_gender_by_movie("data/movie_by_gender/gorinski/")

	# store agarwal dictionary in pickle file
	with open("agarwal_chars_by_gender.pickle", 'wb') as handle:
		pickle.dump(agarwal_chars_by_gender, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# store gorinski dictionary in pickle file
	with open("gorinski_chars_by_gender.pickle", 'wb') as handle:
		pickle.dump(gorinski_chars_by_gender, handle, protocol=pickle.HIGHEST_PROTOCOL)
