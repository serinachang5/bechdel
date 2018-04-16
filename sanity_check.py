import csv
import imdb
import jellyfish as jelly


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

def checkID_agarwal(movies):

	movie_db = imdb.IMDb()
	correct = 0
	incorrect = 0
	id_mismatch = []

	for item in movies:
		movie_by_ID = movie_db.get_movie(item[-1])

		# if levenshtein distance test fails for movie title, continue to check
		# for movie year
		if jelly.levenshtein_distance(str(item[0]), str(movie_by_ID)) >= 10:
			year = str(movie_by_ID["year"])
			writer = list(movie_by_ID["writer"])
			writer_to_str = [str(w) for w in writer]
			with open(item[2]) as fp:
				content = fp.readlines()[:20]
				for w in writer_to_str:
					writer_check = any(w in c for c in content)
				match_year = [s for s in content if year in s]
				if match_year == [] and writer_check == False:
					print("Sanity check failed: \n Year or writer mismatch found. \n {} {}".format(item[-1], item[0]), "\n")
					incorrect += 1
					id_mismatch.append(item)
				else:
					print("Sanity check passed: \n {} {}".format(item[-1], item[0]), "\n")
					correct += 1
		else:
			

			print("Sanity check passed: \n {} {}".format(item[-1], item[0]), "\n")
			correct += 1

	return (str(correct/(correct+incorrect)*100), id_mismatch)

def checkID_gorinski(movies):

	movie_db = imdb.IMDb()
	correct = 0
	incorrect = 0
	id_mismatch = []

	for item in movies:
		movie_by_ID = movie_db.get_movie(item[-1])

		if jelly.levenshtein_distance(str(item[0]), str(movie_by_ID)) >= 15:

			# try:
			# 	with open(item[2]) as fp:
			# 		contents = fp.readlines()[:60]
			# 		for line in contents:
			# 			line = line.strip()
			# 			if len(line) <= 15:
			# IMDB search character not working,
			# no cross match with actor (delete)

			# except FileNotFoundError: --> lots of file mismatches
			# 	id_mismatch.append(item)

			id_mismatch.append(item)
			incorrect += 1

		else:
			correct +=1
	return (str(correct/(correct+incorrect)*100), id_mismatch)

if __name__ == "__main__":

	#agarwal = csv.DictReader(open("agarwal_alignments_with_IDs.csv"))
	gorinski = csv.DictReader(open("gorinski_alignments_with_IDs.csv"))
	samples = getAll(gorinski) 
	#print(samples)
	#all_ = getAll(agarwal)

	#check = checkID_agarwal(all_)
	check = checkID_gorinski(samples)
	print("Sanity check accuracy is: " + check[0])
	print("Movies with incorrect IDs: ", check[1])




