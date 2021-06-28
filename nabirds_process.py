import os
import numpy as np
from collections import defaultdict
import pickle
from sklearn.preprocessing import LabelEncoder



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

    children_chain = list(class_map[:, 0])
    parent_chain = list(class_map[:, 1])
    d = defaultdict(set)
    # top_classes = []
    for i in range(class_map.shape[0]):
        d[class_map[i, 1]].add(class_map[i, 0])
    child_classes = set(class_map[:, 0])
    fine_classes = list(child_classes - d.keys())
    print("num. of fine classes: ", len(fine_classes))
    hier_tree = {}
    levl_count = []
    for fine_class in fine_classes:
        i = 1
        hier_tree[fine_class] = []
        hier_tree[fine_class].append(fine_class)
        #class_super = 100000
        index_1 = children_chain.index(fine_class)
        class_super = parent_chain[index_1]
        while class_super != 0:

            '''index_1 = children_chain.index(fine_class)
            class_super = parent_chain[index_1]
            hier_tree[fine_class].append(class_super)  # lv1
        
            index_2 = children_chain.index(class_super)
            class_super = parent_chain[index_2]
            hier_tree[fine_class].append(class_super)  # lv2
        
            index_3 = children_chain.index(class_super)
            class_super = parent_chain[index_3]
            hier_tree[fine_class].append(class_super)   # lv3'''
            i+=1
            hier_tree[fine_class].append(class_super)
            index = children_chain.index(class_super)
            class_super = parent_chain[index]
        levl_count.append(i)
    return hier_tree, levl_count


# hier_tree[fine_class].append(class_super)

def load_train_test_ids(dataset_path=''):
    """
    Return the image ids of training and testing split
    """

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

    """
    Return full image ids and image paths of NABirds dataset
    """

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

    """
    Return the lists of paths of training-testing split and the corresponding fine-level class labels

    """

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

    # image_ids = []
    image_classes = []
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            # image_id = pieces[0]
            image_class = pieces[1]
            # image_ids.append(image_id)
            image_classes.append(image_class)
    return image_classes


if __name__ == "__main__":

    class_map = save_hierarchy_labels("./nabirds")

    hier_list, lev_count = find_class_hierarchy(class_map)

    print(max(lev_count), min(lev_count))

    train_ids, test_ids = load_train_test_ids("./nabirds")
    print(len(train_ids), len(test_ids), len(train_ids)+len(test_ids))

    image_ids, image_fullpaths = load_full_path_ids("./nabirds")
    print(len(image_ids), len(image_fullpaths))

    image_fine_labels = load_fineclass("./nabirds")

    train_paths, test_paths, train_labels, test_labels = load_split_paths_labels(image_ids, image_fullpaths, train_ids, test_ids, image_fine_labels)

    with open('train_test_split.pickle', 'wb') as f:
        pickle.dump([train_paths, test_paths], f, protocol=pickle.DEFAULT_PROTOCOL)

    hier_np = np.ndarray((len(hier_list.keys()), 4))
    for i, (key, values) in enumerate(hier_list.items()):
        if len(values) == 3:    # hierarchy completion if the fine-level hierarchy is missed
            hier_np[i, 1:] = np.array(values)
            hier_np[i, 0] = hier_np[i, 1]  # the fine-level label is the same as the superclass level
        else:
            hier_np[i, :] = np.array(values)

    fine_list = list(hier_np[:, 0])

    train_labels_np = np.ndarray((len(train_labels), 4))
    test_labels_np = np.ndarray((len(test_labels), 4))

    train_labels = [int(train_label) for train_label in train_labels]  # from string list to int list
    test_labels = [int(test_label) for test_label in test_labels]

    for i, train_label in enumerate(train_labels):
        train_labels_np[i, :] = hier_np[fine_list.index(train_label)]  # from fine-level labels to construct full level labels

    for i, test_label in enumerate(test_labels):
        test_labels_np[i, :] = hier_np[fine_list.index(test_label)]

    for i in range(4):
        print("number of level-%d classes: " %(i+1), np.unique(train_labels_np[:, i]).shape[0])

    label_encoder_list = [LabelEncoder() for i in range(4)]
    # Encode each level of labels to the range of [0, number of classes in corresponding level)
    for i in range(4):
        label_encoder_list[i].fit(list(hier_np[:, i]))

        train_labels_np[:, i] = label_encoder_list[i].transform(train_labels_np[:, i])
        test_labels_np[:, i] = label_encoder_list[i].transform(test_labels_np[:, i])

    np.save("train_labels_nabirds.npy", train_labels_np)
    np.save("test_labels_nabirds.npy", test_labels_np)










