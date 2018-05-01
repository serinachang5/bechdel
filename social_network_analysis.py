from align_gender import get_char_diag_list, get_ff_conversations
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from t3 import get_t3_data
from util import get_data, check_distribution, get_variant_as_key


VERBOSE = True

class Character:
    def __init__(self, name, gender = None):
        self.name = name
        self.gen = gender
        self.adj = set()

    def add_adj(self, adjname):
        self.adj.add(adjname)

class Network:
    def __init__(self):
        self.verts = {}

    def contains(self, cname):
        return cname in self.verts

    def get_vertex(self, cname):
        if not self.contains(cname):
            print(cname, 'is not in the network')
            return None
        return self.verts[cname]

    def add_vertex(self, char):
        self.verts[char.name] = char

    def add_edge(self, cname1, cname2):
        c1 = self.get_vertex(cname1)
        c1.add_adj(cname2)
        c2 = self.get_vertex(cname2)
        c2.add_adj(cname1)

    def print_list(self):
        offset = '   '
        for char in self.verts.values():
            print('Character:', char.name, char.gen)
            for adjname in char.adj:
                print(offset, adjname)

def get_social_network(path, char_dict):
    if 'agarwal' in path:
        source = 'agarwal'
        scenes = get_boundaries_agarwal(path)
    else:
        source = 'gorinski'
        scenes = get_boundaries_gorinski(path)

    var2info = get_variant_as_key(char_dict)

    net = Network()
    for i,scene in enumerate(scenes):
        if VERBOSE: print('Scene', i)
        cdl = get_char_diag_list(scene, var2info, source)
        char_tuples = set([cd[0] for cd in cdl])
        char_tuples = sorted(list(char_tuples), key=lambda x:x[0])  # sort by name
        if VERBOSE: print('Char names:', [x[0] for x in char_tuples])  # print char names
        for i,(cname1, gen1, _) in enumerate(char_tuples):
            if not net.contains(cname1):
                net.add_vertex(Character(cname1, gen1))
            for j,(cname2, gen2, _) in enumerate(char_tuples[i+1:]):
                if not net.contains(cname2):
                    net.add_vertex(Character(cname2, gen2))
                net.add_edge(cname1, cname2)
    return net

X,y = get_t3_data(source='combined')
X = [(x[1], x[2]) for x in X]  # path, char_dict
net = get_social_network(X[10][0], X[10][1])
net.print_list()