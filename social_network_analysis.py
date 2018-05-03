import align_gender as ag
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from t3 import get_t3_data
from util import get_data, check_distribution, get_char_to_lines
import networkx as nx
import matplotlib.pyplot as plt

VERBOSE = False

# make a social network of the characters who have gender scores and > min_lines lines
def get_social_network(path, char_dict, min_lines = 3):
    if 'agarwal' in path:
        source = 'agarwal'
        scenes = get_boundaries_agarwal(path)
    else:
        source = 'gorinski'
        scenes = get_boundaries_gorinski(path)

    var2info = ag.get_variant_as_key(char_dict)
    char2lines = get_char_to_lines(path, char_dict)

    G = nx.Graph()
    for i,scene in enumerate(scenes):
        if VERBOSE: print('Scene', i)
        cdl = ag.get_char_diag_list(scene, var2info, source)
        char_tuples = set([cd[0] for cd in cdl])
        char_tuples = sorted(list(char_tuples), key=lambda x:x[0])  # sort by name
        if VERBOSE: print('Char names:', [x[0] for x in char_tuples])  # print char names
        for i,(cname1, gen1, score1) in enumerate(char_tuples):
            if score1 != 'None' and len(char2lines[cname1]) >= min_lines:
                for j,(cname2, gen2, score2) in enumerate(char_tuples[i+1:]):
                    if score2 != 'None' and len(char2lines[cname2]) >= min_lines:
                        G.add_edge(cname1, cname2)
    return G

def test_network():
    data = get_data()
    ten_things_id = '0147800'
    ten_things = data[ten_things_id]
    path = ten_things[4]
    print(path)
    char_dict = ten_things[5]
    print(char_dict)
    G = get_social_network(path, char_dict)

    cmap = []
    chars = G.nodes()
    for char in chars:
        gen, score, var = char_dict[char]
        if float(score) > .5:
            cmap.append('#FFB6C1') # light pink
        else:
            cmap.append('#87CEFA') # sky blue

    nx.draw_networkx(G, node_size=200, node_color=cmap)
    plt.show()