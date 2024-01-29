import random  # for demo test
import hashlib
import sys

import multiprocessing
import numpy as np
from functools import partial
from queue import Queue
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

splits = 0
parent_splits = 0
fusions = 0
parent_fusions = 0

class Node(object):
    """Base node object. It should be index node
    Each node stores keys and children.
    Attributes:
        parent
    """

    def __init__(self, parent=None):
        """Child nodes are stored in values. Parent nodes simply act as a medium to traverse the tree.
        :type parent: Node"""
        self.keys: list = []
        self.values: list([Node]) = []
        self.parent: Node = parent
        self.digest = None
        #self.center = None
        #self.radius = None
        self.min_key = None
        self.max_key = None


    def calculate_hash(self):
        if type(self.values[0]) is not Node and type(self.values[0]) is not Leaf:

            data_hash = hashlib.sha256((str(self.keys) + str(self.values)).encode()).hexdigest()
            self.digest = data_hash
            return data_hash

        else:

            child_hashes = "".join(str(self.keys))
            for child in self.values:
                child_hash = child.calculate_hash()
                child_hashes += child_hash
            self.digest = hashlib.sha256(child_hashes.encode()).hexdigest()
            return self.digest


    def index(self, key):
        """Return the index where the key should be.
        :type key: str
        """
        for i, item in enumerate(self.keys):
            if key < item:
                return i

        return len(self.keys)

    def __getitem__(self, item):
        return self.values[self.index(item)]

    def __setitem__(self, key, value):
        i = self.index(key)
        self.keys[i:i] = [key]
        self.values.pop(i)
        self.values[i:i] = value


    def split(self):
        """Splits the node into two and stores them as child nodes.
        extract a pivot from the child to be inserted into the keys of the parent.
        @:return key and two children
        """
        global splits, parent_splits
        splits += 1
        parent_splits += 1

        left = Node(self.parent)

        mid = len(self.keys) // 2

        left.keys = self.keys[:mid]
        left.values = self.values[:mid + 1]
        for child in left.values:
            child.parent = left

        key = self.keys[mid]
        self.keys = self.keys[mid + 1:]
        self.values = self.values[mid + 1:]

        return key, [left, self]

    def __delitem__(self, key):
        i = self.index(key)
        del self.values[i]
        if i < len(self.keys):
            del self.keys[i]
        else:
            del self.keys[i - 1]


    def fusion(self):
        global fusions, parent_fusions
        fusions += 1
        parent_fusions += 1

        index = self.parent.index(self.keys[0])
        # merge this node with the next node
        if index < len(self.parent.keys):
            next_node: Node = self.parent.values[index + 1]
            next_node.keys[0:0] = self.keys + [self.parent.keys[index]]
            for child in self.values:
                child.parent = next_node
            next_node.values[0:0] = self.values
        else:  # If self is the last node, merge with prev
            prev: Node = self.parent.values[-2]
            prev.keys += [self.parent.keys[-1]] + self.keys
            for child in self.values:
                child.parent = prev
            prev.values += self.values

    def borrow_key(self, minimum: int):
        index = self.parent.index(self.keys[0])
        if index < len(self.parent.keys):
            next_node: Node = self.parent.values[index + 1]
            if len(next_node.keys) > minimum:
                self.keys += [self.parent.keys[index]]

                borrow_node = next_node.values.pop(0)
                borrow_node.parent = self
                self.values += [borrow_node]
                self.parent.keys[index] = next_node.keys.pop(0)
                return True
        elif index != 0:
            prev: Node = self.parent.values[index - 1]
            if len(prev.keys) > minimum:
                self.keys[0:0] = [self.parent.keys[index - 1]]

                borrow_node = prev.values.pop()
                borrow_node.parent = self
                self.values[0:0] = [borrow_node]
                self.parent.keys[index - 1] = prev.keys.pop()
                return True

        return False


class Leaf(Node):
    def __init__(self, parent=None, prev_node=None, next_node=None):
        """
        Create a new leaf in the leaf link
        :type prev_node: Leaf
        :type next_node: Leaf
        """
        super(Leaf, self).__init__(parent)
        #self.digest = None
        self.next: Leaf = next_node
        if next_node is not None:
            next_node.prev = self
        self.prev: Leaf = prev_node
        if prev_node is not None:
            prev_node.next = self

    def __getitem__(self, item):
        return self.values[self.keys.index(item)]

    def __setitem__(self, key, value):
        i = self.index(key)
        if key not in self.keys:
            self.keys[i:i] = [key]
            self.values[i:i] = [value]
        else:
            self.values[i - 1] = value

    def split(self):
        global splits
        splits += 1

        left = Leaf(self.parent, self.prev, self)
        mid = len(self.keys) // 2

        left.keys = self.keys[:mid]
        left.values = self.values[:mid]

        self.keys: list = self.keys[mid:]
        self.values: list = self.values[mid:]

        # When the leaf node is split, set the parent key to the left-most key of the right child node.
        return self.keys[0], [left, self]

    def __delitem__(self, key):
        i = self.keys.index(key)
        del self.keys[i]
        del self.values[i]

    def fusion(self):
        global fusions
        fusions += 1

        if self.next is not None and self.next.parent == self.parent:
            self.next.keys[0:0] = self.keys
            self.next.values[0:0] = self.values
        else:
            self.prev.keys += self.keys
            self.prev.values += self.values

        if self.next is not None:
            self.next.prev = self.prev
        if self.prev is not None:
            self.prev.next = self.next

    def borrow_key(self, minimum: int):
        index = self.parent.index(self.keys[0])
        if index < len(self.parent.keys) and len(self.next.keys) > minimum:
            self.keys += [self.next.keys.pop(0)]
            self.values += [self.next.values.pop(0)]
            self.parent.keys[index] = self.next.keys[0]
            return True
        elif index != 0 and len(self.prev.keys) > minimum:
            self.keys[0:0] = [self.prev.keys.pop()]
            self.values[0:0] = [self.prev.values.pop()]
            self.parent.keys[index - 1] = self.keys[0]
            return True

        return False


class BPlusTree(object):
    """B+ tree object, consisting of nodes.
    Nodes will automatically be split into two once it is full. When a split occurs, a key will
    'float' upwards and be inserted into the parent node to act as a pivot.
    Attributes:
        maximum (int): The maximum number of keys each node can hold.
    """
    # root: Node

    def __init__(self, maximum=4, type = "key"):
        self.root = Leaf()
        self.maximum: int = maximum if maximum > 2 else 2
        self.minimum: int = self.maximum // 2
        self.depth = 0
        self.type = type

    def cal_memory(self, node):
        if not isinstance(node, (Node, Leaf)):
            return 0

        p_size = sys.getsizeof(node.values[0])
        k_size = sys.getsizeof(node.keys[0])
        min_size = sys.getsizeof(node.min_key)
        max_size = sys.getsizeof(node.max_key)
        h_size = sys.getsizeof(node.digest)

        inc = p_size * len(node.values) + k_size * len(node.keys) + min_size + max_size + h_size + sys.getsizeof(node)

        if isinstance(node, Node):
            for value in node.values:
                inc += self.cal_memory(node=value)

        return inc

    def knum(self, node):
        if not isinstance(node, (Node, Leaf)):
            return 0

        inc = len(node.keys) + 2

        if isinstance(node, Node):
            for value in node.values:
                inc += self.knum(node=value)

        return inc

    def encrypt(self, node, pk):
        if type(node) is not Node and type(node) is not Leaf:
            return

        enc_k = [pk.encrypt(x) for x in node.keys]
        node.keys = enc_k

        node.min_key = pk.encrypt(node.min_key)
        node.max_key = pk.encrypt(node.max_key)

        if type(node) is Node:
            processes = []
            for value in node.values:
                process = multiprocessing.Process(target=self.encrypt, args=(value, pk))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
        else:
            j = 0
            processes = []
            for value in node.values:
                for vector in value:
                    id = vector[0]
                    clu = vector[1]
                    V = vector[2]
                    enc = [pk.encrypt(x) for x in V]
                    node.values[j] = (id, clu, enc)

                process = multiprocessing.Process(target=self.encrypt, args=(value, pk))
                process.start()
                processes.append(process)
                j += 1

            for process in processes:
                process.join()

            node.values = [node.values]

    def encrypt_leaf(self, node, pk):
        if type(node) is not Node and type(node) is not Leaf:
            return

        if type(node) is Node:
            processes = []
            for value in node.values:
                process = multiprocessing.Process(target=self.encrypt_leaf, args=(value, pk))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
        else:
            j = 0
            for value in node.values:
                for vector in value:
                    id = vector[0]
                    clu = vector[1]
                    V = vector[2]
                    enc = [pk.encrypt(x) for x in V]
                    node.values[j] = (id, clu, enc)
                j += 1

            node.values = [node.values]

    def encrypt4(self, node, pk):
        if type(node) is not Node and type(node) is not Leaf:
            return

        enc_k = [pk.encrypt(x) for x in node.keys]
        node.keys = enc_k

        enc_min = pk.encrypt(node.min_key)
        enc_max = pk.encrypt(node.max_key)
        node.min_key = enc_min
        node.max_key = enc_max

        if type(node) is Node:
            for value in node.values:
                self.encrypt4(value, pk)

        else:
            j = 0
            for value in node.values:
                for vector in value:
                    id = vector[0]
                    clu = vector[1]
                    V = vector[2]
                    enc = [pk.encrypt(x) for x in V]
                    node.values[j] = (id, clu, enc)
                j += 1
            node.values = [node.values]

    def find_farthest_point(self, a, q):

        distances = np.linalg.norm(a - q, axis=1)

        farthest_index = np.argmax(distances)

        farthest_point = a[farthest_index]
        farthest_distance = distances[farthest_index]

        return farthest_point, farthest_distance


    def get_ref_key(self, node = None):
        if type(node) is Node or type(node) is Leaf:
            start_node = self.find_leftmost_leaf(node)
            end_node = self.find_rightmost_leaf(node)
            min_key = start_node.keys[0]
            max_key = end_node.keys[-1]
            node.min_key = min_key
            node.max_key = max_key

            for child in node.values:
                self.get_ref_key(child)


    def cluster(self, node = None):
        if type(node) is Node or type(node) is Leaf:
            start_node = self.find_leftmost_leaf(node)
            end_node = self.find_rightmost_leaf(node)
            V = []
            s = start_node
            while s != end_node.next:

                for v in s.values:
                    for vector in v:
                        V.append(vector[2])
                s = s.next

            V = np.array(V)
            V = V.reshape(-1,1024)
            #print(V.shape)


            hull = ConvexHull(V)
            hull_points = V[hull.vertices]
            center = np.mean(hull_points, axis=0)
            radius = np.max(np.linalg.norm(hull_points - center, axis=1))


            kmeans = KMeans(n_clusters=1, random_state=0).fit(V)
            center = kmeans.cluster_centers_[0]
            _,radius = self.find_farthest_point(V, center)

            node.center = center
            node.radius = radius

            for child in node.values:
                self.cluster(child)

    def calculate_hash(self, node = None):
        if type(node.values[0]) is not Node and type(node.values[0]) is not Leaf:

            if self.type == "cluster":
                data_hash = hashlib.sha256((str(node.keys) + str(node.values)+str(node.center)+str(node.radius)).encode()).hexdigest()
            else:
                data_hash = hashlib.sha256((str(node.keys) + str(node.values) + str(node.min_key) + str(node.max_key)).encode()).hexdigest()
            node.digest = data_hash
            return data_hash

        else:

            child_hashes = "".join(str(node.keys))
            for child in node.values:
                child_hash = self.calculate_hash(child)
                child_hashes += child_hash
            if self.type == "cluster":
                child_hashes += str(node.center)+str(node.radius)
            else:
                child_hashes += str(node.min_key) + str(node.max_key)
            node.digest = hashlib.sha256(child_hashes.encode()).hexdigest()
            return node.digest

    def get_hash(self):
        if self.type == "cluster":
            self.cluster(node = self.root)
        else:
            self.get_ref_key(node = self.root)
        return self.calculate_hash(node = self.root)

    def find(self, key) -> Leaf:
        """ find the leaf
        Returns:
            Leaf: the leaf which should have the key
        """
        node = self.root
        # Traverse tree until leaf node is reached.
        while type(node) is not Leaf:
            node = node[key]

        return node

    def __getitem__(self, item):
        return self.find(item)[item]

    def query(self, key):
        """Returns a value for a given key, and None if the key does not exist."""
        leaf = self.find(key)
        return leaf[key] if key in leaf.keys else None

    def change(self, key, value):
        """change the value
        Returns:
            (bool,Leaf): the leaf where the key is. return False if the key does not exist
        """
        leaf = self.find(key)
        if key not in leaf.keys:
            return False, leaf
        else:
            leaf[key] = value
            return True, leaf

    def __setitem__(self, key, value, leaf=None):
        """Inserts a key-value pair after traversing to a leaf node. If the leaf node is full, split
              the leaf node into two.
              """
        if leaf is None:
            leaf = self.find(key)
        leaf[key] = value
        if len(leaf.keys) > self.maximum:
            self.insert_index(*leaf.split())

    def insert(self, key, value):
        """
        Returns:
            (bool,Leaf): the leaf where the key is inserted. return False if already has same key
        """
        leaf = self.find(key)
        if key in leaf.keys:
            # leaf[key].append(value)
            return False, leaf
        else:
            self.__setitem__(key, value, leaf)
            return True, leaf

    def insert_index(self, key, values: list([Node])):
        """For a parent and child node,
                    Insert the values from the child into the values of the parent."""
        parent = values[1].parent
        if parent is None:
            values[0].parent = values[1].parent = self.root = Node()
            self.depth += 1
            self.root.keys = [key]
            self.root.values = values
            return

        parent[key] = values
        # If the node is full, split the  node into two.
        if len(parent.keys) > self.maximum:
            self.insert_index(*parent.split())
        # Once a leaf node is split, it consists of a internal node and two leaf nodes.
        # These need to be re-inserted back into the tree.

    def delete(self, key, node: Node = None):
        if node is None:
            node = self.find(key)
        del node[key]

        if len(node.keys) < self.minimum:
            if node == self.root:
                if len(self.root.keys) == 0 and len(self.root.values) > 0:
                    self.root = self.root.values[0]
                    self.root.parent = None
                    self.depth -= 1
                return

            elif not node.borrow_key(self.minimum):
                node.fusion()
                self.delete(key, node.parent)
        # Change the left-most key in node
        # if i == 0:
        #     node = self
        #     while i == 0:
        #         if node.parent is None:
        #             if len(node.keys) > 0 and node.keys[0] == key:
        #                 node.keys[0] = self.keys[0]
        #             return
        #         node = node.parent
        #         i = node.index(key)
        #
        #     node.keys[i - 1] = self.keys[0]

    def show(self, node=None, file=None, _prefix="", _last=True):
        """Prints the keys at each level."""
        if node is None:
            node = self.root
        print(_prefix, "`- " if _last else "|- ", node.keys, sep="", file=file)
        _prefix += "   " if _last else "|  "

        if type(node) is Node:
            # Recursively print the key of child nodes (if these exist).
            for i, child in enumerate(node.values):
                _last = (i == len(node.values) - 1)
                self.show(child, file, _prefix, _last)

    def output(self):
        return splits, parent_splits, fusions, parent_fusions, self.depth

    def readfile(self, reader):
        i = 0
        for i, line in enumerate(reader):
            s = line.decode().split(maxsplit=1)
            self[s[0]] = s[1]
            if i % 1000 == 0:
                print('Insert ' + str(i) + 'items')
        return i + 1

    def leftmost_leaf(self) -> Leaf:
        node = self.root
        while type(node) is not Leaf:
            node = node.values[0]
        return node

    def find_leftmost_leaf(self,node = None):
        while type(node) is not Leaf:
            node = node.values[0]
        return node

    def find_rightmost_leaf(self,node = None):
        while type(node) is not Leaf:
            node = node.values[-1]
        return node
