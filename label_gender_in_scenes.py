import csv
import os
import pickle

sc_files = set()

# retrieve sample set for movie titles with ids
# sample set - size 10
def getSampleSet(dataset):
	sample = []
	count = 0
	for row in dataset:
		if row['IMDb_id'] != '':
			count += 1
			sample.append((row['Title'], row['Source'], row['File_path'], row['IMDb_id']))
		if count >= 10:
			break
	return sample

# retrieve all movie titles with ids 
def getAll(dataset):
	all_ = []
	for row in dataset:
		if row['IMDb_id'] != '':
			all_.append((row['Title'], row['Source'], row['File_path'], row['IMDb_id']))
	return all_

def get_gender_labels_for_characters(all_movies, char_gender_mapping):

	movie_gender_labels = []
	start_points = []
	start_end = []
	dialogue_boundaries = []
	script_content = []
	relevant_lines = []

	movies_without_gender_labels = []

	for item in all_movies:
		with open(item[2]) as fp:
			try:
				mapping = char_gender_mapping[item[2]]
				contents = fp.readlines()
				for idx, line in enumerate(contents):
					script_content.append(line)
					if line[0:2] == "C|":
						for char in mapping:
							for c in char:
								if c in line:
									movie_gender_labels.append((idx,char[0],line))

			except KeyError:
				movies_without_gender_labels.append(item[2])

	gender_labels = sorted(set(movie_gender_labels), key=movie_gender_labels.index)

	for x in gender_labels:
		start_points.append(x[0])

	for s in range(len(start_points) - 1):
		current_item, next_item = start_points[s], start_points[s + 1]
		start_end.append((current_item, next_item-1))


	for label in gender_labels:
		for bound in start_end:
			if bound[0] == label[0]:
				relevant_lines.append([label[1]]+script_content[bound[0]:bound[1]])
	
	# for item in relevant_lines:
	# 	print(relevant_lines)
	# 	print()


if __name__ == "__main__":

	agarwal = csv.DictReader(open("data/agarwal_alignments_with_IDs_with_bechdel.csv"))
	# gorinski = csv.DictReader(open("data/gorinski_alignments_with_IDs_with_bechdel.csv"))

	with open("agarwal_chars_by_gender.pickle", "rb") as handle:
		a_char_gender = pickle.load(handle)

	all_movies = getSampleSet(agarwal)
	# all_movies = getAll(gorinski)

	m = get_gender_labels_for_characters(all_movies, a_char_gender)
	print(m)

	# for item in all_movies:
	# 	# original_name = item[2][:-4]  agarwal
	# 	original_name = item[2][:-16] # gorinksi
	# 	if not os.path.exists(original_name+item[0]+"_scenes"):
	# 		os.makedirs(original_name+item[0]+"_scenes")

	# 	count = 1
	# 	scenes = get_boundaries_gorinski(item[2])
	# 	for s in scenes:
	# 		output_file = open(original_name+item[0]+"_scenes/"+str(count)+".txt", 'w')
	# 		count +=1
	# 		output_file.writelines(["%s\n" % item for item in s])

	# g_scenes = open("data/gorinski_files_with_scenes.txt", "w")
	# g_scenes.writelines(["%s\n" % item  for item in list(sc_files)])



