# hpdh
[Semantic Hierarchy Preserving Deep Hashing for Large-Scale Image Retrieval](https://arxiv.org/abs/1901.11259)


# CIFAR-100

***cifarDataset.py*** provides the interface to CIFAR-100 dataset (2-level hierarchy, 20 superclasses, 5 subclasses/superclass). This is a modified version of the original 
interface in torchvision. Specifically, it adds coarse (superclass) labels to construct the dataset as an option.  


# [NABirds](https://dl.allaboutbirds.org/nabirds)
NABirds is a dataset with a four-level (exclude the root node, i.e., 'birds') hierarchy. Note that some species in the datasets miss the fine-level class labels. For simplicity, we do the label completion manually by arranging their fine-level labels the same as their parent-level labels. The complete number of classes in each hierarchy is 555, 495, 228, and 22, respectively, from the fine-level to the highest level. 

[Here](https://github.com/cvjena/semantic-embeddings/blob/master/NAB-Hierarchy/hierarchy.svg) is an excellent repository by Björn Barz et al., which provides the visualization of NABirds’ hierarchy.  

# Preprocessing

First, download [NAbirds](https://dl.allaboutbirds.org/nabirds), and assume the directory path is "./nabirds". Then, run the script ***nabirds_process.py***, which does the following:
1) It first extracts the hierarchy information by using all the child-parent label pairs in "hierarchy.txt".
2) Then, it follows the official training-testing split to construct two lists of image paths, which are saved in **train_test_split.pickle**.
3) Thirdly, it stores full-level labels of training and testing splits in two arrays. Each array has a size of (#imags, 4). 
4) Finally, it encodes original class labels in each level to values ranging between \[0, #classes in the corresponding level). The processed labels are saved as NumPy arrays: **train_labels_nabirds.npy** and **test_labels_nabirds.npy**, for training and testing images, respectively.

# Training

To train a model on CIFAR-100 dataset with given code length and model path:

```
python train_hpdh.py --dataset cifar100  --len 48  --path your_model_name 
```

The model will be saved under the directory "./checkpoint" by default. To train the model on NAbirds, change the argument of dataset to **nabirds**.

# Centers updates 

The class centers are updated periodically by the function **centers_update**. The function requires the belonging information of class labels in each child-parent hierarchy. This is achieved by a hierarchical list during training. For example, the two-level hierarchy information of CIFAR-100 is maintained by a list of 20 arrays, where the first array contains 5 fine-level labels, whose parent level labels are coarse label '0'. 

During training, the required hierarchical information will be generated and saved as **labels_rel_cifar100.pickle** and **labels_rel_nabirds.pickle** for future use in CIFAR-100 and NABirds, respectively.

# Evaluation
The evaluation on mean average hierarchical precision (mAHP) and normalized discounted 
cumulative gain (nDCG) requires preparation of configuration files, which computes the ideal cases of AHP/DCG. This is obtained by running ***get_map.py***, which outputs two NumPy files: **[*dataset*]_imAHPs.npy** and **[*dataset*]_iDCGs.npy**, for evaluation.

To evaluate a pre-trained model, add '-e' to turn into evaluation mode:

```
python -e train_hpdh.py --dataset nabirds  --len 48  --path your_model_name 
```

[comment]: <> (# Models: We release the trained models on the NABirds dataset under 32-bit, 48-bit, and 64-bit hashing codes. The compressed .zip file of models can be downloaded here.)
