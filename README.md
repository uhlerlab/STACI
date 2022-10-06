# STACI

Package versions:
- Python 3.10.4
- pytorch 1.9.1
- scanpy 1.9.1
- scipy 1.7.3
- numpy 1.22.4
- scikit-learn 1.1.2
- matplotlib-base 3.5.2
- matplotlib 3.5.2
- seaborn 0.11.2
- pandas 1.4.3
- umap-learn 0.5.3
- anndata 0.8.0
- mnnpy 0.1.9.5 

All neural network models and image processing functions are stored in "image" and "gae".

Training autoencoders and plotting the results can be done in the notebooks listed below, following the instructions provided in the notebooks:
0. Script for computing the adjacency matrices: getXA_starmap.ipynb. A user provided matrix can also be used directly without this step.

1. Training script for graph convolutional autoencoder: train_gae_starmap_multisamples.ipynb

2. Training script for joint latent space: train_jointGAEcnn_starmap_multisamples.ipynb

3. Plotting graph autoencoder's latent space: plotEmbedding_Starmap.ipynb

4. Translation from joint latent space to gene expression: translation_jointCNNgae2Starmap_final.ipynb

6. Nuclei segmentation: segment3D_gpu.ipynb

5. Training script for the regression of plaque size from joint latent space: train_regrsFromJoint_starmap_multisamplesMixed.ipynb

6. Analyzing the results of regression of plaque size from joint latent space: plotRegrsFromJoint_starmap_3Dseg.ipynb


The following notebooks are for generating the validation results we used in our paper and are not necessary for training or analyzing new models:

Validation with 10x Visium data of mouse brain coronal sections: notebooks labeled with "10xADFFPE"
  - The 10x Visium data can be accessed via this link: https://www.10xgenomics.com/resources/datasets/multiomic-integration-neuroscience-application-note-visium-for-ffpe-plus-immunofluorescence-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard
  - The notebooks for training and plotting the underparameterized model with standard batch correction methods (ComBat and MNN) have the additional "extBatchCorrection" label.

Testing overparameterization on scRNA-seq data to remove batch effects: notebooks labeled with "scrnaseq"

Validation on the four held-out STARmap samples: notebooks labeled with "newdata"
