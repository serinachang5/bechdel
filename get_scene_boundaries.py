import csv
import os

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

# define scene boundaries for gorinski movies using :SC: marker
def get_boundaries_gorinski(filepath):
	script_content = []
	scene_points = []
	with open(filepath) as fp:
		contents = fp.readlines()
		for idx, line in enumerate(contents):
			if ":SC:" in line:
				sc_files.add(filepath)
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

if __name__ == "__main__":

	# agarwal = csv.DictReader(open("data/agarwal_alignments_with_IDs_with_bechdel.csv"))
	gorinski = csv.DictReader(open("data/gorinski_alignments_with_IDs_with_bechdel.csv"))

	# all_movies = getAll(agarwal)
	all_movies = getAll(gorinski)

	for item in all_movies:
		# original_name = item[2][:-4]  agarwal
		original_name = item[2][:-16] # gorinksi
		if not os.path.exists(original_name+item[0]+"_scenes"):
			os.makedirs(original_name+item[0]+"_scenes")

		count = 1
		scenes = get_boundaries_gorinski(item[2])
		for s in scenes:
			output_file = open(original_name+item[0]+"_scenes/"+str(count)+".txt", 'w')
			count +=1
			output_file.writelines(["%s\n" % item for item in s])

	g_scenes = open("data/gorinski_files_with_scenes.txt", "w")
	g_scenes.writelines(["%s\n" % item  for item in list(sc_files)])



