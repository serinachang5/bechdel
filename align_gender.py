from csv import DictReader
import imdb
import os
import pickle

'''
Given: screenplay + IMDb id
Return: characters mapped to gender
'''

# return dict: year -> list of name_tuples
def parse_SSA():
    path = './data/names/'
    year_to_names = {}
    for fname in os.listdir(path):
        if fname.startswith('yob'):
            year = fname.rsplit('.txt', 1)[0]
            year = year.split('yob', 1)[1]
            year = int(year)
            print(year)
            name_tuples = get_name_scores(path + fname)
            year_to_names[year] = name_tuples
    return year_to_names

# return dict: name -> score (num of female / sum of male + female)
def get_name_scores(fname):
    name_to_counts = {}
    with open(fname, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            name, gender, count = line.split(',')
            name = name.lower()
            count = int(count)
            if name in name_to_counts:
                if gender == 'F':
                    name_to_counts[name][0] += count
                else:
                    name_to_counts[name][1] += count
            else:
                f_count = m_count = 0.0
                if gender == 'F':
                    f_count = count
                else:
                    m_count = count
                name_to_counts[name] = [f_count, m_count]
    for name, counts in name_to_counts.items():
        score = counts[0]/sum(counts)
        name_to_counts[name] = score
    return name_to_counts

# return dict: chars -> score (avg_score from prior decade)
def classify_chars(screenplay_fname, imdb_id, ssa_dict, db = None):
    if db is None:
        db = imdb.IMDb()
    movie_by_id = db.get_movie(imdb_id)
    movie_year = int(movie_by_id['year'])
    print('Year:', movie_year)
    decade_start = movie_year - 9
    ssa_name_scores = []
    for year in range(decade_start, movie_year+1):
        if year in ssa_dict:
            ssa_name_scores.append(ssa_dict[year])

    movie_chars = parse_agarwal_chars(screenplay_fname)
    char_to_scores = dict((char, [0, 0]) for char in movie_chars)
    for char in movie_chars:
        for year_scores in ssa_name_scores:
            if char in year_scores:
                score = year_scores[char]
                char_to_scores[char][0] += score
                char_to_scores[char][1] += 1
    for char,(score_sum, year_count) in char_to_scores.items():
        if year_count > 0:
            char_to_scores[char] = score_sum/year_count
        else:
            char_to_scores[char] = None
    return char_to_scores

def parse_agarwal_chars(file_path):
    with open(file_path, 'r') as f:
        chars = set()
        for line in f.readlines():
            if line.startswith('C|'):
                # find closed set of prefix titles and remove
                char = line.split()[1]
                char = char.strip('-')
                char = char.lower()
                if not char.startswith('('):
                    chars.add(char)
        return chars

if __name__ == "__main__":
    ssa_dict = pickle.load(open('parsed_ssa.p', 'rb'))
    print('Number of years in SSA parsed:', len(ssa_dict))

    agarwal_csv_fname = 'agarwal_alignments_with_IDs.csv'
    agarwal_reader = DictReader(open(agarwal_csv_fname, 'r'))

    db = imdb.IMDb()
    count = 0
    for row in agarwal_reader:
        if count > 5:
            break
        else:
            count += 1
            path = row['File_path']
            id = row['IMDb_id']
            if id != '':
                print('\n' + row['Title'].upper())
                classifications = classify_chars(path, id, ssa_dict)
                print('Number of characters:', len(classifications))
                for char,score in classifications.items():
                    gender = 'UNK'
                    if score is not None:
                        if score < .1:
                            gender = 'M'
                        elif score > .9:
                            gender = 'F'
                    print(char, gender)