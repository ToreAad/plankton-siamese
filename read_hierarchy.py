from config import plankton_str2int, plankton_int2str

def parse(data, depth, i):
    node = data[i].lstrip()
    children = []
    i += 1
    while i < len(data):
        line = data[i]
        tabs = line.count('\t')
        if tabs <= depth:
            break
        else:
            i, child = parse(data, depth+1, i)
            children.append(child)
    return (i, (node, children))


def get_path(tree, a, path):
    node, children = tree
    path.append(node)
    if node == a:
        return path
    
    for child in children:
        pos_path = get_path(child, a, path)
        if pos_path:
            return pos_path
            
    del path[-1]
    return None


def get_distance(tree, a, b):
    # Following defintion of taxonomic distance given in:
    # http://www.coastalwiki.org/wiki/Measurements_of_biodiversity
    path_to_a = []
    get_path(tree, a, path_to_a)
    path_to_b = []
    get_path(tree, b, path_to_b)

    for i, (p1, p2) in enumerate(zip(path_to_a, path_to_b)):
        if p1 != p2:
            return len(path_to_a) + len(path_to_b) - 2*i
    
    return abs(len(path_to_a) - len(path_to_b))
    

def get_hierarchy():
    with open("hierarchy.txt", "rt") as f:
        data = [line.rstrip() for line in f.readlines()]
        return parse(data, 0, 0)
        

def get_grouping(tree, target_depth):
    parents = {}
    def grouping_helper(tree, depth):
        node, children = tree
        if depth < target_depth:
            if node in plankton_str2int.keys():
                parents[node] = node
            for child in children:
                grouping_helper(child, depth+1)
        else:
            grouping = [node] if node in plankton_str2int.keys() else []
            for child in children:
                grouping += grouping_helper(child, depth+1)
            if depth == target_depth:
                for n in grouping:
                    parents[n] = node
                return
            else:
                return grouping

    grouping_helper(tree, 0)
    children = {}
    for child, parent in parents.items():
        if parent in children:
            children[parent].append(child)
        else:
            children[parent] = [child]
    return parents, children

_, tree = get_hierarchy()
td = {}
for i in range(39):
    for j in range(i, 40):
        a = plankton_int2str[i]
        b = plankton_int2str[j]
        d = get_distance(tree, a, b)
        td[(i,j)] = d
        td[(j,i)] = d
max_distance = max(td.values())

def taxonomic_distance(a, b):
    return td[(a,b)]/max_distance

def taxonomic_ordering(b):
    a = plankton_str2int["egg__other"]
    return taxonomic_distance(a, b)

def taxonomic_grouping(depth):
    parents, _ = get_grouping(tree, depth)
    int_parents = {}

    counter = 0
    get_count = {}
    for key, val in parents.items():
        if not val in get_count:
            get_count[val] = counter
            counter += 1
        int_parents[plankton_str2int[key]] = (get_count[val], val)
    return int_parents

def taxonomic_path(a):
    a = plankton_int2str[a]
    path = []
    get_path(tree, a, path)
    return path

def dfs(tree, dic, counter):
    node, children = tree
    
    dic[node] = counter
    counter += 1
    
    for child in children:
        counter = dfs(child, dic, counter)
    
    return counter


    

if __name__ == "__main__":
    _, tree = get_hierarchy()
    # assert get_distance(tree, "Neoceratium", "Noctiluca") == 7
    # assert get_distance(tree, "Noctiluca", "Tomopteridae") == 2
    # assert get_distance(tree, "Limacidae", "egg__other") == 1

    distances = []
    for i in range(39):
        for j in range(i, 40):
            d = taxonomic_distance(i, j)
            distances.append(d)

    avg = sum(distances)/len(distances)

    for i in range(10):
        test = taxonomic_grouping(i)
