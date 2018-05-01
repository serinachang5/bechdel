# get_scene_boundaries returns individual scenes
# for screenplays and writes them to their 
# respective directories based on the movie name
# for both agarwal and gorinski screenplays

import csv
import os

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

# define scene boundaries for agarwal movies using S| marker
def get_boundaries_agarwal(filepath):

	script_content = []
	scene_points = []
	with open(filepath) as fp:
		contents = fp.readlines()
		for idx, line in enumerate(contents):
			if line[0:2] == "S|":
				scene_points.append(idx)
			script_content.append(line)

	start_end = []
	for s in range(len(scene_points) - 1):
		current_item, next_item = scene_points[s], scene_points[s + 1]
		start_end.append((current_item, next_item-1))

	scene_boundaries = []
	for bound in start_end:
		scene_boundaries.append(script_content[bound[0]:bound[1]])

	return scene_boundaries

# define scene boundaries for gorinski movies using :SC:, EXT., or INT marker
def get_boundaries_gorinski(filepath):

	script_content = []
	scene_points = []
	with open(filepath) as fp:
		contents = fp.readlines()
		for idx, line in enumerate(contents):
			if ":SC:" in line or "EXT." in line or "INT." in line:
				scene_points.append(idx)
			script_content.append(line)

	start_end = []
	for s in range(len(scene_points) - 1):
		current_item, next_item = scene_points[s], scene_points[s + 1]
		start_end.append((current_item, next_item-1))

	scene_boundaries = []
	for bound in start_end:
		scene_boundaries.append(script_content[bound[0]:bound[1]])

	return scene_boundaries

def create_scene_directories(source):

	if source == "agarwal":

		agarwal = csv.DictReader(open("data/agarwal_alignments_with_IDs_with_bechdel.csv"))
		all_movies = getAll(agarwal)

		for item in all_movies:
			original_name = item[2][:-4]
			if not os.path.exists(original_name+"_scenes"):
				os.makedirs(original_name+"_scenes")

			count = 1
			scenes = get_boundaries_gorinski(item[2])
			for s in scenes:
				output_file = open(original_name+"_scenes/"+str(count)+".txt", 'w')
				count +=1
				output_file.writelines(["%s" % item for item in s])

	if source == "gorinski":

		gorinski = csv.DictReader(open("data/gorinski_alignments_with_IDs_with_bechdel.csv"))
		all_movies = getAll(gorinski)

		for item in all_movies:
			original_name = item[2][:-16] 
			sp = original_name.split("/")
			if not os.path.exists(original_name+sp[-2]+"_scenes"):
				os.makedirs(original_name+sp[-2]+"_scenes")

			count = 1
			scenes = get_boundaries_gorinski(item[2])
			for s in scenes:
				output_file = open(original_name+sp[-2]+"_scenes/"+str(count)+".txt", 'w')
				count +=1
				output_file.writelines(["%s" % item for item in s])

if __name__ == "__main__":

	# create_scene_directories("agarwal")
	# create_scene_directories("gorinski")

	