import csv
import os

# retrieve sample set for movie titles with ids
# sample set - size 10
def getSampleSet(agarwal):
	sample = []
	count = 0
	for row in agarwal:
		if row['IMDb_id'] != '':
			count += 1
			sample.append((row['Title'], row['Source'], row['File_path'], row['IMDb_id']))
		if count >= 10:
			break
	return sample

def getAll(agarwal):
	all_ = []
	for row in agarwal:
		if row['IMDb_id'] != '':
			all_.append((row['Title'], row['Source'], row['File_path'], row['IMDb_id']))
	return all_

def get_boundaries(filepath):

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

if __name__ == "__main__":

	agarwal = csv.DictReader(open("data/agarwal_alignments_with_IDs_with_bechdel.csv"))

	all_movies = getAll(agarwal)
	get_boundaries 

	for item in all_movies:
		original_name = item[2][:-4]
		print(original_name)
		if not os.path.exists(original_name+item[0]+"_scenes"):
			os.makedirs(original_name+item[0]+"_scenes")

		count = 1
		scenes = get_boundaries(item[2])
		for s in scenes:
			output_file = open(original_name+item[0]+"_scenes/"+str(count)+".txt", 'w')
			count +=1
			output_file.writelines(["%s\n" % item for item in s])




