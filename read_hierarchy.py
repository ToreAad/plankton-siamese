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
    path_to_a = []
    get_path(tree, a, path_to_a)
    path_to_b = []
    get_path(tree, b, path_to_b)

    depth = 0
    for p1, p2 in zip(path_to_a, path_to_b):
        if p1 != p2:
            return depth
        depth += 1
    
    if path_to_a == path_to_b:
        return depth



        
def get_hierarchy():
    with open("hierarchy.txt", "rt") as f:
        data = [line.rstrip() for line in f.readlines()]
        return parse(data, 0, 0)
        

def get_grouping(tree, target_depth):
    sets = []
    def grouping_helper(tree, depth):
        node, children = tree
        if depth < target_depth:
            if node in plankton_str2int.keys():
                sets.append([node])
            for child in children:
                grouping_helper(child, depth+1)
        else:
            grouping = [node] if node in plankton_str2int.keys() else []
            for child in children:
                grouping += grouping_helper(child, depth+1)
            if depth == target_depth:
                sets.append(grouping)
                return
            else:
                return grouping

    grouping_helper(tree, 0)
    groups = {}
    for i, g in enumerate(sets):
        for n in g:
            groups[n] = i
    return sets, groups

_, tree = get_hierarchy()
def taxonomic_distance(a, b):
    a = plankton_int2str[a]
    b = plankton_int2str[b]
    return get_distance(tree, a, b)

if __name__ == "__main__":
    _, tree = get_hierarchy()
    assert get_distance(tree, "Neoceratium", "Noctiluca") == 7
    assert get_distance(tree, "Noctiluca", "Tomopteridae") == 2
    assert get_distance(tree, "Limacidae", "egg__other") == 1

    sets, groups = get_grouping(tree, 0)
    [print(s) for s in sorted(sets[0])]
    sets, groups = get_grouping(tree, 1)
    sets, groups = get_grouping(tree, 2)
    sets, groups = get_grouping(tree, 3)