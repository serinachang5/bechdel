from csv import DictReader
import os
import pickle

MODE = 'gorinski'

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

# return dict: root -> [gen_score, variants]
def classify_chars(screenplay_fname, movie_year, ssa_dict, mode):
    MALE_PREFIX = ['mr', 'mister', 'duke', 'father', 'brother', 'monsieur', 'prince', 'sir']
    FEMALE_PREFIX = ['ms', 'miss', 'mrs', 'duchess', 'mother', 'sister', 'mademoiselle', 'mlle',
                     'madame', 'princess']
    NEU_PREFIX = ['dr', 'doctor', 'gen', 'general', 'gov', 'governor', 'judge', 'lord', 'major',
                  'master', 'president', 'professor', 'prof', 'senator']
    prefixes = [MALE_PREFIX, FEMALE_PREFIX, NEU_PREFIX]
    # https://mediawiki.middlebury.edu/wiki/LIS/Name_Standards

    decade_start = movie_year - 9
    ssa_name_scores = []
    for year in range(decade_start, movie_year+1):
        if year in ssa_dict:
            ssa_name_scores.append(ssa_dict[year])

    root_to_info = None
    if mode == 'agarwal':
        root_to_info = parse_agarwal_chars(screenplay_fname, prefixes)
    elif mode == 'gorinski':
        root_to_info = parse_gorinski_chars(screenplay_fname, prefixes)

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
def parse_agarwal_chars(file_path, prefixes):
    male_pre, fem_pre, neu_pre = prefixes
    with open(file_path, 'r') as f:
        root_to_info = {} # root mapped to gender score and variants
        for line in f.readlines():
            if line.startswith('C|'):
                # process variant
                var = line.split(None, 1)[1] # cut off C| + white space
                if var.startswith('('): # probably a description e.g. '(QUIETLY)'
                    continue
                var = var.strip() # strip trailing white space

                # process tokens
                toks = var.lower().split()
                if len(toks) > 0:
                    if toks[0].strip('\'\".-:') == 'the': # remove leading 'the'
                        toks.pop(0)
                if len(toks) > 0:
                    first = toks[0].strip('\'\".-:')
                    if first in male_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][0] = 0.0
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [0.0, {var}]
                    elif first in fem_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][0] = 1.0
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [1.0, {var}]
                    elif first in neu_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [None, {var}]
                    else:
                        root = first
                        if root in root_to_info:
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [None, {var}]
        return root_to_info

def parse_gorinski_chars(file_path, prefixes):
    male_pre, fem_pre, neu_pre = prefixes
    with open(file_path, 'r') as f:
        max_ls = 0
        lines = f.readlines()
        for line in lines[:100]: # find max leading space in first 100 lines
            leading_space = len(line) - len(line.lstrip(' '))
            max_ls = max(leading_space, max_ls)

        root_to_info = {} # root mapped to gender score and variants
        for line in lines:
            leading_space = len(line) - len(line.lstrip(' '))
            if leading_space >= max_ls:
                # process variant
                var = line.strip() # strip leading and trailing white space
                if var.startswith('('): # probably a description e.g. '(QUIETLY)'
                    continue

                # process tokens
                toks = var.lower().split()
                if len(toks) > 0:
                    if toks[0].strip('\'\".-:') == 'the': # remove leading 'the'
                        toks.pop(0)
                if len(toks) > 0:
                    first = toks[0].strip('\'\".-:')
                    if first in male_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][0] = 0.0
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [0.0, {var}]
                    elif first in fem_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][0] = 1.0
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [1.0, {var}]
                    elif first in neu_pre:
                        if len(toks) > 1 and not toks[1].startswith('('):
                            root = first + ' ' + toks[1].strip('\'\".-:')
                        else:
                            root = first
                        if root in root_to_info:
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [None, {var}]
                    else:
                        root = first
                        if root in root_to_info:
                            root_to_info[root][1].add(var)
                        else:
                            root_to_info[root] = [None, {var}]
        return root_to_info

def write_to_file(title, str_year, path, root_to_info, save_dir):
    file_name = '_'.join(title.split())
    file_name += '_' + str_year + '.txt'
    found = 0
    with open(save_dir + file_name, 'w') as f:
        f.write(title.upper() + '\n')
        f.write(str_year + '\n')
        f.write(path + '\n')

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

if __name__ == "__main__":
    ssa_dict = pickle.load(open('parsed_ssa.p', 'rb'))
    print('Number of years in SSA parsed:', len(ssa_dict))

    if MODE == 'agarwal':
        reader = DictReader(open('./data/agarwal_alignments_with_IDs_with_bechdel.csv', 'r'))
        save_dir = './movie_by_gender/agarwal/'
    else:
        reader = DictReader(open('./data/gorinski_alignments_with_IDs_with_bechdel.csv', 'r'))
        save_dir = './movie_by_gender/gorinski/'

    count = 0
    for row in reader:
        if row['Bechdel_rating'] != '':
            count += 1
            print('\n' + str(count), row['Title'].upper(), row['Year'])
            root_to_info = classify_chars(row['File_path'], int(row['Year']), ssa_dict, mode=MODE)
            num_assigned = write_to_file(row['Title'], row['Year'], row['File_path'], root_to_info, save_dir)
            print('Number of root names:', len(root_to_info))
            print('Number of gender matches made:', num_assigned)