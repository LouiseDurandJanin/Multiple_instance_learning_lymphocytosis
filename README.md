# Multi Instance Learning for Lymphocytosis classification

Final project for the Deep Learning for Medical Imaging course from MVA master, supervised by Maria Vakalopoulou (CentraleSupelec) and Olivier Colliot(CNRS). 

**Abstract**

  *Lymphocytosis manifests as an elevation in lymphocyte levels within the bloodstream. Detecting the underlying cause of this condition involves conducting a comprehensive analysis, which includes examining blood smear images and accompanying clinical data like age and lymphocyte concentration. This approach aids in pinpointing the specific disease responsible for the observed lymphocytosis. We propose a method that melds insights from two different data sources - images and clinical attributes - using minimal computational resources for diagnosing lymphocytosis within a multi-instance learning framework. Initially, we focus on extracting image features through self-supervised learning techniques. This enables us to condense each image into a compact, low-dimensional representation.In the next phase, we integrate information derived from these image embeddings alongside clinical attributes to diagnose lymphocytosis effectively.*

**Description**

* data.py : Data preparation and augmentation
* autoencoder.py : CNN-based to extract relevant features on images
* model.py : Final model with attention-based pooling and binary classifier
* train_feature_extractor.py : Training script for the autoencoder
* train_final_model.py : Training script for the final model
