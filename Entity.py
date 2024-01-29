import csv
import sys
import hashlib
import numpy as np
from query import Query
from encrypt import AESEncrypt,RSA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
from bPlusTree import Node,Leaf,BPlusTree
from sklearn.cluster import KMeans
from pympler import asizeof

class  VO:
    def __init__(self):
        self.digests = None
        #self.vectors = None
        self.VR = None
        self.sig_hroot = None
        self.sig_R = None
        self.sig_svMMR = None
        self.modality = None
        self.keys = None

    def set_data(self,digests = None, VR = None, sig_hroot = None, sig_R = None, sig_svMMR = None, modality = None, keys = None):
        self.digests = digests
        #self.vectors = vectors
        self.VR = VR
        self.sig_hroot = sig_hroot
        self.sig_R = sig_R
        self.sig_svMMR = sig_svMMR
        self.modality = modality
        self.keys = keys

    def cal_memory(self):

        d_size = sys.getsizeof(self.digests)
        vr_size = sys.getsizeof(self.VR)
        hroot_size = sys.getsizeof(self.sig_hroot)
        R_size = sys.getsizeof(self.sig_R)
        svMMR_size = sys.getsizeof(self.sig_svMMR)
        modality_size = sys.getsizeof(self.modality)
        keys_size = sys.getsizeof(self.keys)

        return d_size+vr_size+hroot_size+R_size+svMMR_size+modality_size+keys_size

class DAP:
    def __init__(self, sk):
        self.sk = sk



class ServiceProvider:
    def __init__(self, img_knn = None, txt_knn = None, top_k = 10, n_clusters = 9, img_feature = None, txt_feature = None):
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.img_knn = img_knn
        self.txt_knn = txt_knn

    def query_process(self):
        img_ids = []
        txt_ids = []
        VR = []
        keys = []
        if self.img_feature is not None:
            nNeighbors = self.txt_knn.kNeighbors(self.img_feature)
            for i in range(self.top_k):
                #print(nNeighbors.answerSet[i])
                txt_ids.append(nNeighbors.answerSet[i][0])
                index = nNeighbors.answerSet[i][1]
                key = index * self.txt_knn.c + np.linalg.norm(self.txt_knn.o[index].vect - nNeighbors.answerSet[i][2])
                keys.append(key)
                VR.append(nNeighbors.answerSet[i][2])

        if self.txt_feature is not None:
            nNeighbors = self.img_knn.kNeighbors(self.txt_feature)
            for i in range(self.top_k):
                #print(nNeighbors.answerSet[i])
                img_ids.append(nNeighbors.answerSet[i][0])
                index = nNeighbors.answerSet[i][1]
                key = index * self.img_knn.c + np.linalg.norm(self.img_knn.o[index].vect - nNeighbors.answerSet[i][2])
                keys.append(key)
                VR.append(nNeighbors.answerSet[i][2])

        return img_ids, txt_ids, keys, VR

    def vo_generate(self, ids, VR, root, q, sig_svMMR, rsa, keys, typ):
        VR = np.array(VR)
        VR = VR.reshape(-1, 1024)

        VR_center = q
        _, VR_radius = self.find_farthest_point(VR, VR_center)

        sig_R = []
        node_stop = []

        #self.traverse(root, digest = digests, vectors = vectors, VR_center = VR_center, VR_radius = VR_radius, VR = VR)

        subtree = Node()
        if typ == "cluster":
            self.traverse_and_build_subtree(root, subtree, VR_center, VR_radius, node_stop)
        else:
            self.traverse_and_build_subtree_by_key(root, subtree, keys, node_stop)
        digests = subtree.values[0]

        if self.img_feature is not None:
            modality = "text"
            for i in ids:
                with open(f'./signatures/texts/{i}.txt', 'r') as sig_file:
                    sig = sig_file.read()
                    sig_R.append(sig)
        else:
            modality = "image"
            for i in ids:
                with open(f'./signatures/images/{i}.txt', 'r') as sig_file:
                    sig = sig_file.read()
                    sig_R.append(sig)

        #vectors = np.array(vectors)
        #vectors = vectors.reshape(-1, 1024)
        #print(vectors.shape)

        sig_hroot = rsa.sign_digest(root.digest.encode('utf-8'))
        vo = VO()
        vo.set_data(digests, VR, sig_hroot, sig_R, sig_svMMR, modality, keys)
        return vo, node_stop

    def traverse2(self, root, digest, vectors, VR_center, VR_radius, VR):
        if type(root) is Node or type(root) is Leaf:
            dist = np.linalg.norm(VR_center - root.center)
            if dist >= VR_radius+root.radius:
                digest.append(root.digest)
                parent = root.parent
                right_sibling = None
                for i, value in parent.values:
                    if value == root and i < len(parent.values)-1:
                        right_sibling = parent.values[i+1]
                        break

                self.traverse(right_sibling, digest, vectors, VR_center, VR_radius, VR)
            else:
                if type(root) is Leaf:
                    digest.append(root.digest)
                    for v in root.values:
                        # print(v[0][2])
                        for vector in v[0][2]:
                            vectors.append(vector)
                else:
                    for child in root.values:
                        self.traverse(child, digest, vectors, VR_center, VR_radius, VR)

    def traverse(self, root, digest, vectors, VR_center, VR_radius, VR):
        if root is None:
            return

        if type(root) is Node or type(root) is Leaf:
            if self.intersects(root.center, root.radius, VR_center, VR_radius):
                digest.append(root)
                if type(root) is Leaf:
                    for v in root.values:
                        for vector in v[0][2]:
                            vectors.append(vector)
                elif type(root) is Node:
                    for child in root.values:

                        if self.intersects(child.center, child.radius, VR_center, VR_radius):
                            self.traverse(child, digest, vectors, VR_center, VR_radius, VR)

    def traverse_and_build_subtree_by_key(self, root, newtree_root, keys, node_stop):
        if root is None:
            return

        if isinstance(root, Node) or isinstance(root, Leaf):
            if self.is_range_present(keys, root.min_key, root.max_key):
                new_node = Node() if isinstance(root, Node) else Leaf()
                new_node.keys = root.keys
                new_node.values = root.values
                new_node.digest = root.digest
                new_node.min_key = root.min_key
                new_node.max_key = root.max_key
                new_node.parent = newtree_root
                newtree_root.values.append(new_node)

                if isinstance(root, Node):
                    for child in root.values:

                        self.traverse_and_build_subtree_by_key(child, newtree_root, keys, node_stop)
        else:
            node_stop.append(root)

    def traverse_and_build_subtree(self, root, newtree_root, VR_center, VR_radius, node_stop):
        if root is None:
            return

        if isinstance(root, Node) or isinstance(root, Leaf):
            if self.intersects(root.center, root.radius, VR_center, VR_radius):
                new_node = Node() if isinstance(root, Node) else Leaf()
                new_node.keys = root.keys
                new_node.values = root.values
                new_node.digest = root.digest
                new_node.center = root.center
                new_node.radius = root.radius
                new_node.parent = newtree_root
                newtree_root.values.append(new_node)

                if isinstance(root, Node):
                    for child in root.values:

                        self.traverse_and_build_subtree(child, newtree_root, VR_center, VR_radius, node_stop)
            else:
                node_stop.append(root)

    def intersects(self, node_center, node_radius, VR_center, VR_radius):
        dist = np.linalg.norm(VR_center - node_center)
        return dist < VR_radius + node_radius

    def is_range_present(self, lst, min_val, max_val):
        for num in lst:
            if min_val <= num <= max_val:
                return True
        return False

    def find_farthest_point(self, a, q):
        distances = np.linalg.norm(a - q, axis=1)
        farthest_index = np.argmax(distances)
        farthest_point = a[farthest_index]
        farthest_distance = distances[farthest_index]

        return farthest_point, farthest_distance


class DataOwner:
    def __init__(self):
        self.Ks = None
        self.img_knn = None
        self.txt_knn = None
        self.img_digest = None
        self.txt_digest = None

    def build_index(self,top_k, n_clusters, img_vectors, txt_vectors, n_node,type):
        img_knn = Query(top_k)
        img_digest = img_knn.fit(n_clusters, img_vectors, n_node=n_node, type = type)
        txt_knn = Query(top_k)
        txt_digest = txt_knn.fit(n_clusters, txt_vectors, n_node=n_node, type = type)
        self.img_knn = img_knn
        self.txt_knn = txt_knn
        self.img_digest = img_digest
        self.txt_digest = txt_digest

    def sign_data(self):
        decrypted_image_prefix = './decrypted_data/images/'
        decrypted_txt_prefix = './decrypted_data/texts/'
        rsa = RSA()
        with open('D:/PyCharmFile/MultiModalRetrieval/wikipedia_dataset/trainset_txt_img_cat.list', 'r') as file:
            lines = file.readlines()
            for index, line in enumerate(lines):
                img = line.split("\t")[1]
                txt = line.split("\t")[0]

                img_path = decrypted_image_prefix + str(img) + ".jpg"
                with open(img_path, 'rb') as file:
                    image_data = file.read()

                txt_path = decrypted_txt_prefix + str(txt) + ".txt"
                with open(txt_path, 'r', encoding="UTF-8") as file:
                    txt_data = file.read()

                img_sig = rsa.sign_image(image_data)
                txt_sig = rsa.sign_text(txt_data)

                img_sig_hex = img_sig.hex()
                txt_sig_hex = txt_sig.hex()

                with open(f'./signatures/images/{index}.txt', 'w') as img_sig_file:
                    img_sig_file.write(img_sig_hex)

                with open(f'./signatures/texts/{index}.txt', 'w') as txt_sig_file:
                    txt_sig_file.write(txt_sig_hex)
        return rsa


class Client:
    def __init__(self, img_feature = None, txt_feature = None, K = None):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.K = K

    def process_data(self,data):
        img_feature = []
        txt_feature = []

    def decrypt(self,R):
        return R

    def verify(self, vo, q, ids, rsa , type, node_stop):
        max_dist = np.linalg.norm(q - vo.VR[-1])
        vectors = []

        #check_dist = np.all(max_dist < np.linalg.norm(q - vo.vectors, axis=1))

        check_hroot = True
        check_svMMR = True
        check_sig_R = True
        check_dist = True

        hroot = self.calculate_hash(vo.digests, vectors, vo.VR, ids, type, vo.keys, node_stop)
        if rsa.verify_digest_signature(hroot.encode('utf-8'), vo.sig_hroot) is False:
            check_hroot = False

        print(len(vectors))

        check_dist = np.all(max_dist < np.linalg.norm(q - vectors, axis=1))

        decrypted_image_prefix = './decrypted_data/images/'
        decrypted_txt_prefix = './decrypted_data/texts/'

        with open('D:/PyCharmFile/MultiModalRetrieval/wikipedia_dataset/trainset_txt_img_cat.list', 'r') as file:
            lines = file.readlines()
            if vo.modality == "image":
                for i in range(len(ids)):
                    img = lines[ids[i]].split("\t")[1]
                    img_path = decrypted_image_prefix + str(img) + ".jpg"
                    with open(img_path, 'rb') as file:
                        image_data = file.read()

                    img_sig = bytes.fromhex(vo.sig_R[i])
                    if rsa.verify_image_signature(image_data, img_sig) is False:
                        check_sig_R = False
                        break

            else:
                for i in range(len(ids)):
                    txt = lines[ids[i]].split("\t")[0]
                    txt_path = decrypted_txt_prefix + str(txt) + ".txt"
                    with open(txt_path, 'r', encoding="UTF-8") as file:
                        txt_data = file.read()

                    txt_sig = bytes.fromhex(vo.sig_R[i])
                    if rsa.verify_text_signature(txt_data, txt_sig) is False:
                        check_sig_R = False
                        break

        if check_dist and check_sig_R and check_svMMR and check_hroot:
            return True
        else:
            return False

    def calculate_hash(self, node, vectors, VR, ids, typ, keys, node_stop):

        if type(node) is Leaf:
            #data_hash = hashlib.sha256((str(node.keys) + str(node.values)+str(node.center)+str(node.radius)).encode()).hexdigest()
            if len(node.values) > 0 and self.is_range_present(keys, node.keys[0], node.keys[-1]):
                for v in node.values:
                    for vector in v:
                        if vector[0] not in ids:
                            vectors.append(vector[2])
            return node.digest

        if type(node) is Node:
            if node in node_stop:
                return node.digest
            else:
                child_hashes = "".join(str(node.keys))
                for child in node.values:

                    child_hash = self.calculate_hash(child, vectors, VR, ids, typ, keys, node_stop)
                    child_hashes += child_hash
                if typ == "cluster":
                    child_hashes += str(node.center)+str(node.radius)
                else:
                    child_hashes += str(node.min_key) + str(node.max_key)
                data_hash = hashlib.sha256(child_hashes.encode()).hexdigest()
                return data_hash

    def is_range_present(self, lst, min_val, max_val):
        for num in lst:
            if min_val <= num <= max_val:
                return True
        return False

def get_datalist(file_name):
    data_list = []

    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            dt1 = row[0]
            dt2 = row[1][3:-3]
            dt2 = dt2.split(",")
            dt2 = [float(x) for x in dt2]
            #dt2 = [float(x) for x in dt2[: 512]]
            #data = (dt1,dt2)
            data_list.append(dt2)
    return data_list

if __name__ == '__main__':

    DO = DataOwner()
    top_k = 1
    n_clusters = 5
    n_node = 3
    type_tree = "key"

    datalist1 = get_datalist("./wikipedia_vectors/img.csv")
    datalist2 = get_datalist("./wikipedia_vectors/txt.csv")

    img_vectors = np.array(datalist1).astype(float)
    txt_vectors = np.array(datalist2).astype(float)

    DO.build_index(top_k, n_clusters, img_vectors, txt_vectors, n_node, type_tree)
    rsa = DO.sign_data()
    print(rsa)

    img_feature = np.random.uniform(0, 1, (1, 1024))
    SP = ServiceProvider(img_knn = DO.img_knn, txt_knn = DO.txt_knn, top_k = top_k, n_clusters = n_clusters, img_feature = img_feature, txt_feature = None)
    img_ids, txt_ids, keys, VR = SP.query_process()

    ids = txt_ids
    root = DO.txt_knn.bplustree.root

    q = img_feature
    vo, node_stop = SP.vo_generate(ids = ids, VR = VR, root = root, q = q, sig_svMMR=1, rsa = rsa, keys = keys, typ = type_tree)

    client = Client()
    verification = client.verify(vo, q, ids, rsa, type_tree, node_stop)
    print(verification)
    memory = vo.cal_memory()
    print(memory/(1024))