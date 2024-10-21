# Vseq2Human

## VAbout

This work, **Vseq2Human**, combines Deep Learning and Monte Carlo Simulation to predict human-infecting viruses using genomic data. Predictions were evaluated by aggregating probability estimates from multiple runs. Class 1 (human-infecting) entries were confidently classified if at least 75% of predictions exceeded 0.5 (Q1 > 0.5), while class 0 (non-human-infecting) entries were assigned when most predictions were below 0.5 (Q3 < 0.5).


## Requirements:

To run this project, you will need Python (version X.X or higher) and the following libraries.

## Dependencies

- **PyBioMed**:A Python library used for generating various biological sequence-based descriptors(GetDAC, GetPseDNC, GetPseKNC, GetSCPseDNC).
- **iLearn**: A toolkit for generating numerical feature descriptors(CKSNAP, DNC, RCKmer, TNC) from biological sequences.


## Usage

1. Clone the repository.
2. Install the required dependencies listed above.
3. Run the prediction script by providing the genomic sequences for analysis.

## Output

The output will provide a classification indicating whether a virus is predicted to infect humans (Class 1) or not (Class 0), along with the corresponding probability estimates from multiple Monte Carlo simulation runs.

## For more information:

Please check the article (It will be available soon).

## Credit/Acknowledgment

This research was supported by **Universidad de Las Américas (UDLA)** (Quito, Ecuador).
