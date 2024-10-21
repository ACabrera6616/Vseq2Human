# Vseq2Human

## VAbout

This project combines Deep Learning and Monte Carlo Simulation to predict human-infecting viruses using genomic data. Predictions were evaluated by aggregating probability estimates from multiple runs. Class 1 (human-infecting) entries were confidently classified if at least 75% of predictions exceeded 0.5 (Q1 > 0.5), while class 0 (non-human-infecting) entries were assigned when most predictions were below 0.5 (Q3 < 0.5).


## Requirements:

## Dependencies

    - PyBioMed: A Python library used for generating various biological sequence-based descriptors(GetDAC, GetPseDNC, GetPseKNC, GetSCPseDNC).
    - iLearn: A toolkit for generating numerical feature descriptors(CKSNAP, DNC, RCKmer, TNC) from biological sequences.

## For more information:

Please check the article (It will be available soon).

## Credit/Acknowledgment

This research was supported by Universidad de Las Américas (Quito, Ecuador).
