# hpdh
[Semantic Hierarchy Preserving Deep Hashing for Large-Scale Image Retrieval](https://arxiv.org/abs/1901.11259)


# CIFAR-100

***cifarDataset.py*** provides the interface to CIFAR-100 dataset (2-level hierarchy, 20 superclasses, 5 subclasses/superclass). This is a modified version of the original 
interface in torchvision. Specifically, it adds coarse (superclass) labels to construct the dataset as an option.  


# [NABirds](https://dl.allaboutbirds.org/nabirds)
NABirds is a dataset with a four-level (exclude the root node, i.e., 'birds', which does not provide any information gain) hierarchy. Note that some species in the datasets miss the fine-level class labels. For simplicity, we do the label completion manually by using the same fine-level class labels as their parent-level labels. After processing, the number of classes in each hierarchy level is 555, 495, 228, and 22, respectively, from the fine-level to the highest level. 

[Here](https://github.com/cvjena/semantic-embeddings/blob/master/NAB-Hierarchy/hierarchy.svg) is an excellent repository by Björn Barz et al., which provides the visualization of NABirds’ hierarchy.  

# Preprocessing

First, download [NAbirds](https://dl.allaboutbirds.org/nabirds), and assume the directory path is "./nabirds". Then, run the script ***nabirds_process.py***, which does the following things:
1) It first extracts the hierarchy information of the dataset by using all the child-parent label pairs in "hierarchy.txt".
2) Then, it follows the official training-testing split to construct image paths, and the training and testing image paths are saved in the pickle file **train_test_split.pickle**
3) Thirdly, according to the extracted label information, it builds NumPy array to store full-level class labels of the training and testing images, respectively. The array is of size (#imags, 4). 
4) Finally, it encodes each level of class labels to values ranging between \[0, #classes in the corresponding level). The processed labels are saved as numpy arrays **train_labels_nabirds.npy** and **test_labels_nabirds.npy**, respectively, for training and testing images.

# Training

To train a model on CIFAR-100 dataset with given code length and model path:

```
python train_hpdh.py --dataset cifar100  --len 48  --path your_model_name 
```

The model will be saved under the directory "./checkpoint" by default. To train the model on NAbirds, simply change the argument of dataset to **nabirds**.

# Centers updates 

The class centers in each level hierarchy are updated periodically performed by the function **centers_update**. The function requires the belonging information, which describes the class labels of each child-parent level. This is achieved by a hierarchical list during training. For example, the two-level hierarchy information of CIFAR-100 is maintained by a list of five arrays, where the first array contains 20 fine-level labels, whose parent level labels are coarse label '0'. 

When running ***train_hpdh.py*** for training, the required hierarchical information will be generated automatically and saved as **labels_rel_cifar100.pickle** and **labels_rel_nabirds.pickle** for the CIFAR-100 and NABirds datasets, respectively.

# Evaluation
The evaluation on mean average Hierarchical Precision (mAHP) and normalized Discounted 
Cumulative Gain (nDCG) requires preparation of the configuration files, which computes the ideal cases of AHP/DCG. This can be obtained by running 
***get_map.py***, which generates two NumPy files: **[*dataset*]_imAHPs.npy** and **[*dataset*]_iDCGs.npy** for evaluation.

To evaluate a trained model on the NABirds dataset with given code length and mode path, simply turn into evaluation mode:

```
python -e train_hpdh.py --dataset nabirds  --len 48  --path your_model_name 
```

[comment]: <> (# Models: We release the trained models on the NABirds dataset under 32-bit, 48-bit, and 64-bit hashing codes. The compressed .zip file of models can be downloaded here.)
