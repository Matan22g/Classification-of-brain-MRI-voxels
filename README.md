# Classification-of-brain-MRI-voxels
Implementation of the backpropagation algorithm for fully connected neural network with the apply of it for the task of brain MRI voxels classification, when voxel can belong or not to multiple sclerosis lesion.

# Intro

Multiple Sclerosis is one of the most common non-traumatic neurological diseases in
young adults. It is a chronic inflammatory disease in which the immune system attacks the
central nervous system and damages myelin, myelin producing cells and underlying nerve
fibers.
Multiple sclerosis lesions appear as hyper-intense regions in FLAIR MRI modality
images:

![Screenshot 2021-07-05 111625](https://user-images.githubusercontent.com/64740256/124439756-7b5b4c80-dd82-11eb-9cc9-62be9644f178.png)


# Dateset
Dataset for this exercise contains patches of 32 x 32 pixels. These patches are extracted
from the axial view of FLAIR MRI modality. Patch considered to be positive if the central
pixel (x=16, y=16) belongs to multiple sclerosis lesion, and negative if the central pixel
belongs to the healthy region of the brain. The dataset is already divided to
train and validation and needs to be extracted from the zip file.


# Results

![lr_0 4_batch_size_32_epochs_63_hidden_3072](https://user-images.githubusercontent.com/64740256/124439653-5666d980-dd82-11eb-8f79-70af39c686d6.jpg)
