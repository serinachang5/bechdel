import align_gender as ag
import os
import numpy as np
import random

def get_data(source = 'combined'):
    path = './movie_by_gender/'

    if source == 'agarwal' or source == 'gorinski':
        id_to_info = {}
        for fname in os.listdir(path + source + '/'):
            id, info = ag.parse_by_gender_file(path + source + '/' + fname)
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

def error_analysis_of_char_names():
    data = get_data()
    data = list(data.values())
    for i,info in enumerate(random.sample(data, 10)):
        print('Sample',i)
        char_dict = info[5]
        var_dict = ag.get_variant_as_key(char_dict)
        for var in var_dict:
            if len(var.split()) > 1:  # more than one word
                if '\'S' in var:
                    print(var)


if __name__ == "__main__":
    # combined_data = get_data()
    # print(check_distribution(combined_data))
    error_analysis_of_char_names()