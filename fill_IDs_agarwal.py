import csv
import imdb
import jellyfish as jelly




def getData(agarwal):

	data = []

	for row in agarwal:
		data.append([row["Title"], row["Source"], row["File_path"], row["IMDb_id"]])
	return data

def getMissingID(data):

	res = []

	movie_db = imdb.IMDb()

	for item in data:
		if item[0] != '' and item[-1] == '':
			movie_by_title = movie_db.search_movie(item[0])
			first = movie_by_title[0:1]
			#print(first[0].movieID)
			res.append((item[0], item[1], item[2], first[0].movieID))
		else:
			res.append((item[0], item[1], item[2], item[3]))

	with open("agarwal_alignments_with_IDs.csv", "w") as fin:
		writer = csv.writer(fin, delimiter=",")
		writer.writerow(["Title", "Source", "File_path", "IMDb_id"])
		writer.writerows(res)
			#print(first[0])
			#movie_by_title_to_str = [str(m) for m in list(movie_by_title)]
			#print(movie_by_title_to_str)
			# with open(item[2]) as fp:
			# 	content = fp.readlines()[:30]
			# 	print(content)






			# print(row["Title"])
			# movie_by_title = movie_db.search_movie(row['Title'])
			# print(movie_by_title)
			# print()




if __name__ == "__main__":

	agarwal = csv.DictReader(open("cleaned_agarwal_alignments.csv"))
	all_ = getData(agarwal)
	#print(all_)
	ids = getMissingID(all_)
	print(ids)
	# #samples = getSampleSet(agarwal) 
	# all_ = getAll(agarwal)

	# check = checkID(all_)
	# print("Sanity check accuracy is: " + check[0])
	# print("Movies with incorrect IDs: ", check[1])




