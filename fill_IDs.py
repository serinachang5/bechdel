import csv
import imdb
import jellyfish as jelly


def getData(filename):

	data = []

	for row in filename:
		data.append([row["Title"], row["Source"], row["File_path"], row["IMDb_id"]])
	return data

def getMissingID(data, result_filename):

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

	with open(result_filename, "w") as fin:
		writer = csv.writer(fin, delimiter=",")
		writer.writerow(["Title", "Source", "File_path", "IMDb_id"])
		writer.writerows(res)


if __name__ == "__main__":

	#agarwal = csv.DictReader(open("cleaned_agarwal_alignments.csv"))
	gorinski = csv.DictReader(open("gorinski_alignments.csv"))
	all_ = getData(gorinski)
	#ids = getMissingID(all_, "agarwal_alignments_with_IDs.csv")
	ids = getMissingID(all_, "gorinski_alignments_with_IDs.csv")




