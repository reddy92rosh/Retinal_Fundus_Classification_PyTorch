Retinal Fundus Image Classification Using
PyTorch
1 Introduction
A simple image classification task using retinal fundus image dataset [1] containing
7673 images belonging to 4 classes, namely normal (N), cataract (C),
proliferative diabetic retinopathy (PDR) and glaucoma (G).
2 Dataset
The dataset used in this work was made available through the Kaggle community,
specifically Retinal Fundus Images [1]. It consists of 11 different classes of
pathologies, out of which I use 4 classes (normal, cataract, proliferative diabetic
retinopathy and glaucoma) for training the convolutional neural network (CNN)
model in this work. The dataset is split into the training, validation and test
subsets as shown in the Table 1.
The image size varies from 512 x 512 to 1024 x 1024. To train the classification
CNN model, they were resized to 224 x 224. We can see some of the examples
of these fundus images in Fig. 1 and Fig. 2.
Table 1: Number of images of each class in the training, validation and test set,
respectively.
Pathology Train Validation Test
Normal 2641 54 179
Cataract 1369 24 112
Proliferative DR 1295 30 91
Glaucoma 1678 44 156
3 Methodology
The model used to train this image classification task was VGG11 made publicly
available at [2]. The networks are trained by randomly augmenting the fundus
images on-the-fly using a series of translation, rotation, flipping and gamma
correction operations. The networks are trained using the SGD optimizer with a
learning rate of 0.01, a momentum of 0.9 for 100 epochs and batch size of 32 on
a machine equipped with NVIDIA RTX 2080 Ti GPU with 11GB of memory.
The VGG11 model was trained using cross entropy loss as objective function.
2
Fig. 1: Examples of Normal (top) and Cataract (bottom) retinal fundus images,
respectively.
3
Fig. 2: Examples of Proliferative Diabetic Retinopathy (top) and Glaucoma (bottom)
retinal fundus images, respectively.
4
3.1 Dependencies
– Python3
– Numpy, Pandas, Scikit-learn, ImageIO, OpenCV
– Pytorch, PIL
3.2 How to run
– Download and install the dataset from [1].
– Use .csv files in the csvs folder and edit the image file location in them.
– To train, edit lines 17 to 26 as required in trainModel.py and run: python
trainModel.py
– To evaluate, edit lines 17 to 23 as required in testModel.py and run: python
testModel.py
4 Results
– Training Time: 206 seconds per epoch
– Inference Time: 13 seconds on the test dataset
– Average Accuracy on the test dataset: 0.981
– Average F1 Score on the test dataset: 0.961
The training results are given in the ’train001.csv’ file in the trainedModels
folder above. The per class accuracy, sensitivity, specificity and F1 score are
shown in Table 2.
Table 2: Per-class evaluation metrics on the test set.
Pathology Accuracy Sensitivity Specificity F1 Score
Normal 0.998 1.00 0.997 0.997
Cataract 0.968 0.955 0.971 0.926
Proliferative DR 0.996 1.00 0.995 0.989
Glaucoma 0.962 0.903 0.986 0.933
References
1. K S Sanjay Nithish: Retinal Fundus Images,
https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images
2. Sathyan, A.: Pytorch-image-classification. https://github.com/anilsathyan7/pytorchimage-
classification (2019)
