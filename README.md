# Improved Gesture Recognition based on sEMG Signals and TCN

This is the code accompaniment for the following paper presented at [ICASSP 2019](https://2019.ieeeicassp.org/): <br/>
P. Tsinganos et al., “Improved Gesture Recognition Based on sEMG Signals and TCN,” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 1169–1173.

## Requirements
The following python packages are needed to run the code:
- keras 2.2.4 (from tensorflow library)
- tensorflow 1.13.1
- sklearn 0.20.3
- scipy 1.2.1
- numpy 1.16.2

## Usage
To replicate the experiments described in the paper run: `bash run.sh`. Before running the code download the Ninapro dataset as described in the [dataset](../blob/master/dataset/README.md) folder.

## License
If this code helps your research, please cite the [paper](https://ieeexplore.ieee.org/document/8683239/).

```
@inproceedings{Tsinganos2019,
address = {Brighton, UK},
author = {Tsinganos, Panagiotis and Cornelis, Bruno and Cornelis, Jan and Jansen, Bart and Skodras, Athanassios},
booktitle = {ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
doi = {10.1109/ICASSP.2019.8683239},
month = {may},
pages = {1169--1173},
publisher = {IEEE},
title = {{Improved Gesture Recognition Based on sEMG Signals and TCN}},
year = {2019}
}
```

### Acknowledgements
The work is supported by the "Andreas Mentzelopoulos Scholarships for the University of Patras" and the VUB-UPatras International
Joint Research Group (IJRG) on ICT.


### Contact Details
Panagiotis Tsinganos | PhD Candidate  
University of Patras, Greece  
Vrije Universiteit Brussel, Belgium  
<panagiotis.tsinganos@ece.upatras.gr>
