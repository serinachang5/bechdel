from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski

from csv import DictReader
import os
import pickle

'''ALIGN GENDERS'''
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
def get_name_scores(ssa_fname):
    name_to_counts = {}
    with open(ssa_fname, 'r') as f:
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

# transform line into variant
def line_to_variant(line, source):
    assert(source == 'agarwal' or source == 'gorinski')
    if source == 'agarwal':
        var = line.split(None, 1)[1] # cut off C| + white space
        var = var.strip()  # strip trailing white space
    else:  # source == gorinski
        var = line.strip()
    return var

# transform variant into root and return gender score if prefix found
def variant_to_root(var):
    MALE_PREFIX = ['mr', 'mister', 'duke', 'father', 'brother', 'monsieur', 'prince', 'sir']
    FEMALE_PREFIX = ['ms', 'miss', 'mrs', 'duchess', 'mother', 'sister', 'mademoiselle', 'mlle',
                     'madame', 'princess']
    NEU_PREFIX = ['dr', 'doctor', 'gen', 'general', 'gov', 'governor', 'judge', 'lord', 'major',
                  'master', 'president', 'professor', 'prof', 'senator']
    # https://mediawiki.middlebury.edu/wiki/LIS/Name_Standards

    root = None
    gen_score = None
    toks = var.lower().split()
    # if len(toks) > 0:
        # if toks[0].strip('\'\".-:') == 'the': # remove leading 'the'
            # toks.pop(0)
    if len(toks) > 0:
        first = toks[0].strip('\'\".-:')
        if first in MALE_PREFIX:
            if len(toks) > 1 and not toks[1].startswith('('):
                root = first + ' ' + toks[1].strip('\'\".-:')
            else:
                root = first
            gen_score = 0.0
        elif first in FEMALE_PREFIX:
            if len(toks) > 1 and not toks[1].startswith('('):
                root = first + ' ' + toks[1].strip('\'\".-:')
            else:
                root = first
            gen_score = 1.0
        elif first in NEU_PREFIX:
            if len(toks) > 1 and not toks[1].startswith('('):
                root = first + ' ' + toks[1].strip('\'\".-:')
            else:
                root = first
        else:
            root = first
    return root, gen_score

# return dict: root -> [gen_score, variants]
def classify_chars(screenplay_fname, movie_year, ssa_dict, source):
    decade_start = movie_year - 9
    ssa_name_scores = []
    for year in range(decade_start, movie_year+1):
        if year in ssa_dict:
            ssa_name_scores.append(ssa_dict[year])

    root_to_info = None
    if source == 'agarwal':
        root_to_info = parse_agarwal_chars(screenplay_fname)
    elif source == 'gorinski':
        root_to_info = parse_gorinski_chars(screenplay_fname)

    for root,(gen, vars) in root_to_info.items():
        if gen is None:
            sum_score = 0
            year_count = 0
            for year_scores in ssa_name_scores:
                if root in year_scores:
                    score = year_scores[root]
                    sum_score += score
                    year_count += 1
            if year_count > 0:
                root_to_info[root][0] = sum_score/year_count

    return root_to_info

# return dict: root -> gender, variants
def parse_agarwal_chars(file_path):
    with open(file_path, 'r') as f:
        root_to_info = {} # root mapped to gender score and variants
        for line in f.readlines():
            if line.startswith('C|'):
                var = line_to_variant(line, source='agarwal')
                if var.startswith('('): # probably a description e.g. '(QUIETLY)'
                    continue

                root, gen_score = variant_to_root(var)
                if root not in root_to_info:
                    root_to_info[root] = [None, set()]
                if gen_score is not None:
                    root_to_info[root][0] = gen_score
                root_to_info[root][1].add(var)

    return root_to_info

# return dict: root -> gender, variants
def parse_gorinski_chars(file_path):
    with open(file_path, 'r') as f:
        max_ls = 0
        lines = f.readlines()
        for line in lines[:100]:  # find max leading space in first 100 lines
            leading_space = len(line) - len(line.lstrip(' '))
            max_ls = max(leading_space, max_ls)

        root_to_info = {} # root mapped to gender score and variants
        for line in lines:
            leading_space = len(line) - len(line.lstrip(' '))
            if leading_space >= max_ls:
                var = line_to_variant(line, source='gorinski')
                if var.startswith('('): # probably a description e.g. '(QUIETLY)'
                    continue

                root, gen_score = variant_to_root(var)
                if root not in root_to_info:
                    root_to_info[root] = [None, set()]
                if gen_score is not None:
                    root_to_info[root][0] = gen_score
                root_to_info[root][1].add(var)

    return root_to_info

# write char_dict and other metadata to file
def write_to_file(row, root_to_info, save_dir):
    file_name = '_'.join(row['Title'].split())
    file_name += '_' + row['Year'] + '.txt'
    found = 0
    with open(save_dir + file_name, 'w') as f:
        f.write(row['Title'].upper() + '\n')
        f.write('Year: ' + row['Year'] + '\n')
        f.write('IMDb id: ' + row['IMDb_id'] + '\n')
        f.write('Bechdel score: ' + row['Bechdel_rating'] + '\n')
        f.write('File path: ' + row['File_path'] + '\n')

        f.write('\nCHARACTERS\n')
        offset = '   '

        for root, (gen_score, variants) in root_to_info.items():
            f.write('Root: ' + root + '\n')
            gender = 'UNK'
            if gen_score is not None:
                gen_score = round(gen_score, 4)
                if gen_score < .1:
                    gender = 'M'
                    found += 1
                elif gen_score > .9:
                    gender = 'F'
                    found += 1
            f.write(offset + 'Gender: ' + gender + '\n')
            f.write(offset + 'Score: ' + str(gen_score) + '\n')
            f.write(offset + 'Variants:\n')

            for v in variants:
                f.write((offset * 2) + v + '\n')

    return found

# returns title, year, imdb_id, bechdel, path, char_dict
def parse_by_gender_file(path_to_file):
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        title = lines[0].strip()
        year = lines[1].strip().replace('Year: ', '')
        imdb_id = lines[2].strip().replace('IMDb id: ', '')
        bechdel = lines[3].strip().replace('Bechdel score: ', '')
        path = lines[4].strip().replace('File path: ', '')

        char_dict = {}
        line_idx = 7  # 5 is \n, 6 is CHARACTERS
        while line_idx < len(lines):
            root = lines[line_idx].strip().replace('Root: ', '')
            line_idx += 1
            gender = lines[line_idx].strip().replace('Gender: ', '')
            line_idx += 1
            score = lines[line_idx].strip().replace('Score: ', '')
            line_idx += 2 # skip Variants:
            variants = []
            while line_idx < len(lines) and not lines[line_idx].strip().startswith('Root: '):
                variant = lines[line_idx].strip()
                if len(variant) > 0:
                    variants.append(variant)
                line_idx += 1
            char_dict[root] = [gender, score, variants]
        info = [title, year, imdb_id, bechdel, path, char_dict]
    return imdb_id, info


'''PARSE SCENE'''''
def get_variant_as_key(char_dict):
    var2info = {}
    for char,(gen, score, variants) in char_dict.items():
        for v in variants:
            var2info[v] = (char, gen, score)
    return var2info

# given a scene, return the list of char-dialogue tuples where char = (name, gender, score)
def get_char_diag_list(scene, var2info, source):
    char_diag_list = []

    if source == 'agarwal':
        idx = 0
        while idx < len(scene):
            if scene[idx].startswith('C|'):
                var = line_to_variant(scene[idx], source='agarwal')
                if var.startswith('('):  # probably a description e.g. '(QUIETLY)'
                    idx += 1
                    continue

                if var in var2info:
                    curr_char = var2info[var]  # root, score
                    idx += 1
                    diag = []
                    while idx < len(scene) and scene[idx].startswith('D|'):
                        line = scene[idx].split(None, 1)[1]  # cut off D| + white space
                        line = line.strip()
                        diag.append(line)
                        idx += 1
                    char_diag_list.append((curr_char, diag))
                else:
                    idx += 1
            else:
                idx += 1

    else:  # source == 'gorinski'
        idx = 0
        while idx < len(scene):
            var = line_to_variant(scene[idx], source='gorinski')
            if var.startswith('('):
                idx += 1
                continue
            if var in var2info:
                curr_char = var2info[var]
                idx += 1
                if idx < len(scene):
                    diag = []
                    diag_ls = len(scene[idx]) - len(scene[idx].lstrip(' '))
                    while idx < len(scene) and diag_ls == len(scene[idx]) - len(scene[idx].lstrip(' ')):
                        diag.append(scene[idx].strip())
                        idx += 1
                    char_diag_list.append((curr_char, diag))
            else:
                idx += 1

    return char_diag_list

# given a char-dialogue list, return the continuous ff conversations
def get_ff_conversations(char_diag_list):
    ffs = []
    prev_char = ''
    prev_score = -1
    prev_line = ''
    idx = 0
    while idx < len(char_diag_list):
        (char, gen, score), diag = char_diag_list[idx]
        if score != 'None' and float(score) > .5 and \
                prev_score != 'None' and float(prev_score) > .5 and prev_char != char:
            ff = [(prev_char, prev_line), (char, ' '.join(diag))]
            idx += 1
            # include any continuous dialogue from only these female characters
            while idx < len(char_diag_list) and (char_diag_list[idx][0][0] == prev_char or char_diag_list[idx][0][0] == char):
                ff.append((char_diag_list[idx][0][0], ' '.join(char_diag_list[idx][1])))
                idx += 1
            ffs.append(ff)
        # either it wasn't a second female character so we're on the same line
        # or it was a second female character and we went through their conversation
        # and exited because it's not one of those characters anymore or idx == len
        if idx < len(char_diag_list):
            (char, gen, score), diag = char_diag_list[idx]
            prev_char = char
            prev_score = score
            prev_line = ' '.join(diag)
            idx += 1
    return ffs

def test_funcs_on_agarwal():
    # 1. parse by_gender file
    ten_things_by_gender = './movie_by_gender/agarwal/10_things_i_hate_about_you_1999.txt'
    imdb_id, info = parse_by_gender_file(ten_things_by_gender)
    # print(info)

    # 2. set up
    title, year, imdb_id, bechdel, path, char_dict = info
    scenes = get_boundaries_agarwal(path)
    # print(scenes[:5])
    var_dict = get_variant_as_key(char_dict)
    # print(var_dict)

    # 3. parse scene
    for sc in scenes[:1]:
        print('SCENE')
        print(sc)
        cdl = get_char_diag_list(sc, var_dict, source='agarwal')
        print('Char-Diag List')
        for cd in cdl:
            print(cd)
        ffs = get_ff_conversations(cdl)
        print('Fem-Fem dialogues')
        for ff in ffs:
            print(ff)

if __name__ == "__main__":
    # SOURCE = 'agarwal'
    #
    # ssa_dict = pickle.load(open('parsed_ssa.p', 'rb'))
    # print('Number of years in SSA parsed:', len(ssa_dict))
    #
    # reader = DictReader(open('./data/' + SOURCE + '_alignments_with_IDs_with_bechdel.csv', 'r'))
    # save_dir = './movie_by_gender/' + SOURCE + '/'
    #
    # count = 0
    # for row in reader:
    #     if row['Bechdel_rating'] != '':
    #         count += 1
    #         print('\n' + str(count), row['Title'].upper(), row['Year'])
    #         root_to_info = classify_chars(row['File_path'], int(row['Year']), ssa_dict, source=SOURCE)
    #         num_assigned = write_to_file(row, root_to_info, save_dir)
    #         print('Number of root names:', len(root_to_info))
    #         print('Number of gender matches made:', num_assigned)
    test_funcs_on_agarwal()



