# Vseq2Human

## About

This work, **Vseq2Human**, combines Deep Learning and Monte Carlo Simulation to predict human-infecting viruses using genomic data. Predictions were evaluated by aggregating probability estimates from multiple runs. Class 1 (human-infecting) entries were confidently classified if at least 75% of predictions exceeded 0.5 (Q1 > 0.5), while class 0 (non-human-infecting) entries were assigned when most predictions were below 0.5 (Q3 < 0.5).


## Requirements:

To run this project, you will need Python (version 3.8.16) and the following libraries.

- biopython==1.83
- Keras==2.4.1
- Keras-Preprocessing==1.1.2
- joblib==1.2.0
- numpy==1.19.5
- PyBioMed==1.0
- scikit-learn==1.2.2
- tensorflow==2.4.1

## Dependencies

- **PyBioMed:** A Python library used for generating various biological sequence-based descriptors(GetDAC, GetPseDNC, GetPseKNC, GetSCPseDNC).
- **iLearn**: A [toolkit](https://github.com/Superzchen/iLearn) for generating numerical feature descriptors (CKSNAP, DNC, RCKmer, TNC) from biological sequences.

## Usage

1. Clone the repository.
2. Install the required dependencies listed above.
3. Run the prediction script by providing the genomic sequences for analysis.

## Datasets:

Datasets were extracted from the Virus-Host and Virus Pathogen databases for training and validating the deep learning model. Then, A set of Molecular descriptors (from PyBioMed and iLearn) were computed to evaluate the similarity of the viral genomes. The final number of genomes was obtained with a cutoff of 0.001, and the resulting datasets used for built the models are in the following files

- **TRAINING_Disimilarity_0.001_py:** Traning Data.
- **EXTERNAL_Disimilarity_0.001_py:** Test Data.
- **Sample_Virus_Disimilarity_0_001** External validatation dataset

## Overview

The provided code "DeepMC.py" train a model Vseq2Human which combines Deep Learning and Monte Carlo to predict whether viruses can infect humans based on genomic data.

## Output

The output will provide a classification indicating whether a virus is predicted to infect humans or not, along with the corresponding probability estimates from multiple Monte Carlo simulation runs.

## For more information:

Please check the article (It will be available soon).

## Credit/Acknowledgment

This research was supported by **Universidad de Las Américas (UDLA)** (Quito, Ecuador).
