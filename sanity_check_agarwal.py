import csv
import imdb
import jellyfish as jelly

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


def checkID(samples):

	movie_db = imdb.IMDb()

	correct = 0
	incorrect = 0

	for item in samples:
		movie_by_ID = movie_db.get_movie(item[-1])

		# if levenshtein distance test fails for movie title, continue to check
		# for movie year
		if jelly.levenshtein_distance(str(item[0]), str(movie_by_ID)) >= 10:
			year = str(movie_by_ID["year"])
			with open(item[2]) as fp:
				content = fp.readlines()[:20]
				match_year = [s for s in content if year in s]
				if match_year == []:
					print("Sanity check failed: \n Year mismatch found. \n {} {}".format(item[-1], item[0]), "\n")
					incorrect += 1
				else:
					print("Sanity check passed: \n {} {}".format(item[-1], item[0]), "\n")
					correct += 1
					#find_idx = match_year[0].find(year)
					#print(len(match_year[0][find_idx:]))
		else:
			print("Sanity check passed: \n {} {}".format(item[-1], item[0]), "\n")
			correct += 1

	return str(correct/(correct+incorrect)*100)

if __name__ == "__main__":

	agarwal = csv.DictReader(open("agarwal_alignments.csv"))
	samples = getSampleSet(agarwal) 
	print("Sanity check accuracy is: " + checkID(samples))