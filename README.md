# Spec2Tax: Predicting sample organism taxonomy using machine learning on repository-scale untargeted mass spectrometry data

## How to use
Classifier.prepare_datasets.ipynb prepares datasets from the spec2vec embeddings of the data from GNPS. The output of this notebook is four pairs of data (300-dimensional spec2vec embeddings of the raw mass spec data) and label files, named X and y respectively. The interclass dataset is placed in a directory called "class". The intraclass datasets are placed in directories named after the class ("Mammalia", "Gammaproteobacteria", and "Magnoliopsida"). 

Classifier_inter_and_intraclass_classification.ipynb assumes there are datasets at the paths from the "prepare_datasets" notebook, and trains and tests an elastic net model with a logistic regression estimator. It saves the results from the tested model in a "results" directory in the dataset's directory. 

**Running as a docker**
"definitions", "src", ".dockerignore", Dockerfile, requirements.txt, and Makefile were used to run the training and testing steps via AWS Sagemaker.

## Datasets

**Interclass Classification**
This dataset samples 5000 spectra from each of these 5 taxonomic classes: Mammalia, Actinomycetia, Coscinodiscophyceae, Anthozoa, Insecta. The spectra were converted to a 300-dimension embedding by the spec2vec model. Failed conversions (amounting to less than <~2% of the total dataset) were excluded. The datasets are saved in as numpy arrays in X (spec2vec embeddings) and y (class labels converted to ints of the embeddings) pickle files. 

**Intraclass Mammalia Classification**
These dataset samples 5000 spectra from each of these 3 taxonomic families within the class Mammalia: Hominidae, Muridae, Rhinocerotidae. Data processing and filenames are as described above. 

**Intraclass Gammaproteobactera Classification**
These dataset samples 1000 spectra from each of these 2 taxonomic families within the class Gammaproteobactera: Enterobacteriaceae and Morganellaceae. Data processing and filenames are as described above. 

**Intraclass Magnoliopsida Classification**
These dataset samples 500 spectra from each of these 3 taxonomic families within the class Magnoliopsida: Malvaceae, Euphorbiaceae and Moraceae. Data processing and filenames are as described above. 
