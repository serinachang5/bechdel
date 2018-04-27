import csv
import os

def get_gender_by_movie():

	all_files = os.listdir("data/movie_by_gender/agarwal")

	count = 0

	gender_in_movies_agarwal = {}

	for filename in all_files:

		chars = []
		char_info = []
		start_end_char = []
		char_boundaries = []
		store_gender = []
		store_variants = []
		concat = []


		with open("data/movie_by_gender/agarwal/"+filename) as fp:
			if count < 3:
				contents = fp.readlines()
				gender_in_movies_agarwal[contents[4][11:]] = []
				for idx, line in enumerate(contents):
					if line[0:5] == "Root:":
						chars.append(idx)
					char_info.append(line)
			count += 1


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
			
				gender_in_movies_agarwal[contents[4][11:]] = new_concat

	return gender_in_movies_agarwal
			



if __name__ == "__main__":

	agarwal_movie_genders = get_gender_by_movie()
	print(agarwal_movie_genders)





