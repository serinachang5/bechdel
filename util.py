from align_gender import parse_by_gender_file
import os
import numpy as np

def get_data(source = 'combined'):
    path = './movie_by_gender/'

    if source == 'agarwal' or source == 'gorinski':
        id_to_info = {}
        for fname in os.listdir(path + source + '/'):
            id, info = parse_by_gender_file(path + source + '/' + fname)
            id_to_info[id] = info
        return id_to_info

    elif source == 'combined':
        aga = get_data(source='agarwal')
        gor =  get_data(source='gorinski')
        combined = aga
        overlap = 0
        for id in gor:
            if id not in combined:
                combined[id] = gor[id]
            else:
                overlap += 1
        return combined

def check_distribution(data, test = None):
    num_class = 4 if test is None else 2
    counts = np.zeros(num_class)
    for movie_id, info in data.items():
        title, year, imdb_id, bechdel, path, char_dict = info
        score = int(bechdel)
        if test is None:
            counts[score] += 1
        else:
            label = 1 if score >= test else 0
            counts[label] += 1
    return counts


if __name__ == "__main__":
    combined_data = get_data()
    print(check_distribution(combined_data))