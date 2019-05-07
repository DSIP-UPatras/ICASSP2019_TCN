Download and extract the [Ninapro-DB1](http://ninapro.hevs.ch/data1) dataset (login required) into the Ninapro-DB1 folder. 
Then run `python create_dataset.py` to generate the preprocessed data used for training and testing.
The structure of this folder should be:

    .
    ├── Ninapro-DB1/
    |   ├── S1_A1_E1.mat
    |   ├── ...
    |
    ├── Ninapro-DB1-Proc/
    |   ├── subject-01/
    |   |   ├── gesture-00/
    |   |   |   ├── rms/
    |   |   |       ├── rep-00.mat
    |   |   |       ├── ...
    |   |   ├── ...
    |   ├── ...
    |
    ├── create_dataset.py
    └── README.md

