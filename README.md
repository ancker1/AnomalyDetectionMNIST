# AnomalyDetectionMNIST

emanc16@student.sdu.dk  
 
This repository contains the work performed during a Project in Artificial Intelligence at the University of Southern Denmark.  
  
In this repository, implementation and evaluation of several methods for anomaly detection in the MNIST dataset can be found.  
## Notebooks
A list of the most relevant files and their contents is presented.  

**AE-GMM.ipynb**: utilizes a deep hybrid model based on an autoencoder trained on the MNIST dataset and used as a feature extractor for a GMM that is being used as an outlier detector.  
**Autoencoder.ipynb**: utilizes a simply fully-connected autoencoder structure directly as an anomaly detector by using reconstruction loss as outlier score.  
**ConvAE.ipynb**: utilizes a convolutional autoencoder structure directly as an anomaly detector by using the reconstruction loss as outlier score.  
**DeepAutoencoder.ipynb**: similar to Autoencoder.ipynb but uses a fully-connected autoencoder structure with several layers.  
**Traditional.ipynb**: uses LOF and kNN directly as anomaly detectors.  
**VGG16-GMM.ipynb**: Deep hybrid model using the VGG16 network pre-trained on ImageNet as deep feature extractor. GMM is used as an anomaly detector on the extracted features.  
**VarAE.ipynb**: uses a variational fully-connected autoencoder directly as an anomaly detector by using the reconstruction loss as outlier score.  