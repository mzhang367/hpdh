import os
import numpy as np
#import pdb
from collections import defaultdict
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

def list_images(basePath, contains=None):
    image_types = (".jpg")
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def save_hierarchy_labels(dirname):
    """
    save the labels in txt file to numpy file
    """
    fid = open(os.path.join(dirname, "hierarchy.txt"))
    lines = fid.readlines()
    num_lines = len(lines)
    class_map = np.ndarray((num_lines, 2), dtype=np.uint)

    for i, line in enumerate(lines):
        line = line.strip('\n')
        line_split = line.split(' ')
        class_map[i, 0], class_map[i, 1] = line_split[0], line_split[1]

    return class_map


def find_class_hierarchy(class_map):

    # Extract full hierarchy label information from all the child-parent class label pairs
    # child_chain = list(class_map[:, 0])
    child_nodes = list(class_map[:, 0])  # child nodes are naturally unique
    parent_chain = list(class_map[:, 1])
    # d = defaultdict(set)
    parent_nodes = set(class_map[:, 1])   # create unique parent nodes
    # for i in range(class_map.shape[0]):
        # d[class_map[i, 1]].add(class_map[i, 0])
    # child_classes = set(class_map[:, 0])  # create unique child node
    fine_classes = list(set(child_nodes) - parent_nodes)  # remove all the nodes which are parent nodes to obtain leaf nodes
    print("num. of fine classes: ", len(fine_classes))
    hier_tree = {}
    levl_count = []
    # we now trace back from leaf nodes to the root node
    for fine_class in fine_classes:
        i = 1
        hier_tree[fine_class] = []
        hier_tree[fine_class].append(fine_class)
        index_1 = child_nodes.index(fine_class)
        class_super = parent_chain[index_1]     # find a valid route
        while class_super != 0:     # not the root node (bird)
            i += 1  # tree depth +1
            hier_tree[fine_class].append(class_super)   # add the superclass label to the corresponding list of fine-level class
            index = child_nodes.index(class_super)  # since we have not reach the root node, we trace the next parent node
            class_super = parent_chain[index]
        levl_count.append(i)    # record the tree depth (eliminate root node) from each leaf rode to root node
    return hier_tree, levl_count



def load_train_test_ids(dataset_path=''):
  train_ids = []
  test_ids = []

  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train:
        train_ids.append(image_id)
      else:
        test_ids.append(image_id)

  return train_ids, test_ids


def load_full_path_ids(dataset_path):

    root_path = os.path.join(dataset_path, 'images')

    image_ids = []
    image_fullpaths = []
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            image_path = pieces[1]
            image_fullpath = os.path.join(root_path, image_path)
            image_ids.append(image_id)
            image_fullpaths.append(image_fullpath)
    return image_ids, image_fullpaths


def load_split_paths_labels(full_ids, full_paths, train_ids, test_ids, full_labels):

    train_paths = []
    test_paths = []
    train_class_list = []
    test_class_list = []

    for train_id in train_ids:
        train_index = full_ids.index(train_id)
        train_path = full_paths[train_index]
        train_class = full_labels[train_index]

        train_paths.append(train_path)
        train_class_list.append(train_class)

    for test_id in test_ids:
        test_index = full_ids.index(test_id)
        test_path = full_paths[test_index]
        test_class = full_labels[test_index]

        test_paths.append(test_path)
        test_class_list.append(test_class)

    return train_paths, test_paths, train_class_list, test_class_list


def load_fineclass(dataset_path=''):

    image_classes = []
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_class = pieces[1]
            image_classes.append(image_class)
    return image_classes




if __name__ == "__main__":

    class_map = save_hierarchy_labels("./nabirds")
    hier_tree, lev_count = find_class_hierarchy(class_map)
    with open('nabirds_hierarchy.pickle', 'wb') as f:
        pickle.dump(hier_tree, f, protocol=pickle.DEFAULT_PROTOCOL)
    # print(len(hier_tree.keys()))
    print(max(lev_count), min(lev_count))

    train_ids, test_ids = load_train_test_ids("./nabirds")
    # print(len(train_ids), len(test_ids), len(train_ids)+len(test_ids))

    image_ids, image_fullpaths = load_full_path_ids("./nabirds")
    # print(len(image_ids), len(image_fullpaths))

    image_fine_labels = load_fineclass("./nabirds")

    train_paths, test_paths, train_labels, test_labels = load_split_paths_labels(image_ids, image_fullpaths, train_ids, test_ids, image_fine_labels)

    with open('train_test_nabirds.pickle', 'wb') as f:
        pickle.dump([train_paths, test_paths, train_labels, test_labels], f, protocol=pickle.DEFAULT_PROTOCOL)

    '''
    with open('hier_dict.pickle', 'rb') as f:
        hier_dict = pickle.load(f)
    five_hiers = 0
    four_hiers = 0
    print(len(hier_dict.keys()))
    hier_dict_np = np.ndarray((555, 4))
    for i, (key, values) in enumerate(hier_dict.items()):
        # print(len(values))
        
        if len(values) == 3:
            # print(values)
            hier_dict_np[i, 1:] = np.array(values)
            hier_dict_np[i, 0] = hier_dict_np[i, 1]
        else:
            hier_dict_np[i, :] = np.array(values)
        # print(five_hiers, four_hiers)
    np.save("hier_dict_np.npy", hier_dict_np)'''

    '''hier_dict = np.load("hier_dict_np.npy")
    fine_list = list(hier_dict[:, 0])

    train_labels_np = np.ndarray((len(train_labels), 4))
    test_labels_np = np.ndarray((len(test_labels), 4))

    train_labels = [int(train_label) for train_label in train_labels]
    test_labels = [int(test_label) for test_label in test_labels]

    for i, train_label in enumerate(train_labels):
        train_labels_np[i, :] = hier_dict[fine_list.index(train_label)]

    for i, test_label in enumerate(test_labels):
        test_labels_np[i, :] = hier_dict[fine_list.index(test_label)]

    label_encoder_list = [LabelEncoder() for i in range(4)]
    for i in range(4):
        label_encoder_list[i].fit(list(hier_dict[:, i]))

        train_labels_np[:, i] = label_encoder_list[i].transform(train_labels_np[:, i])
        test_labels_np[:, i] = label_encoder_list[i].transform(test_labels_np[:, i])

    np.save("train_labels_birds.npy", train_labels_np)
    np.save("test_labels_birds.npy", test_labels_np)

    with open('label_encoders.pickle', 'wb') as f:
        pickle.dump(label_encoder_list, f, protocol=pickle.HIGHEST_PROTOCOL)'''

    '''train_labels = np.load("train_labels_birds.npy")
    for i in range(4):
        train_set = np.unique(train_labels[:, i])
        print(train_set.shape)'''










