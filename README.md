# quantum-jets

Classical SVM vs quantum kernel SVM on jet data from the ATLAS Open Data.
Code for the ICAI'26 paper.

Each event gets a label: 1 if the leading jet is well reconstructed (reco pT / truth pT
between 0.9 and 1.1), 0 otherwise. The features are pt, eta, phi and m of the three
leading jets plus the number of jets and HT, 14 in total. PCA reduces them to 4, 6 and 8
components, and for the quantum kernel each component is encoded on one qubit
(ZZ feature map, statevector simulation with Qiskit Aer).

## Data

The file `mc_jets-2020.part01.root` (about 2 GB) from CERN Open Data record:
https://opendata.cern.ch/record/15010

Put it in the repository root or pass the path with `--path`.

ATLAS simulated samples collection for jet reconstruction training, as part of the 2020 Open Data release.
CERN Open Data Portal. DOI: 10.7483/OPENDATA.ATLAS.L806.5CKU

## Installation

Python 3.10 or newer. The file is read with uproot.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Usage

```
python -m qjets
```

Some options:

```
python -m qjets --kernel all          # RBF and linear SVM
python -m qjets --qubit-dims 4 6      # which PCA dimensions to run
python -m qjets --n-samples 10000     # size of the balanced sample
python -m qjets --device CPU          # CPU, GPU or auto (auto is the default and picks the CPU)
```

All options: `python -m qjets --help`

Each run creates a folder `runs/<date>-<time>-<config hash>/` with the config, package
versions, stage timings, results and the log.

## Example results

Default settings: 1000 events (500 per class), 80/20 split, seed 42. ROC-AUC on the
test set.

| qubits | classical SVM (RBF) | quantum kernel SVM | mean off-diagonal kernel value | 2^-n |
|---|---|---|---|---|
| 4 | 0.89 | 0.72 | 0.091 | 0.063 |
| 6 | 0.89 | 0.58 | 0.029 | 0.016 |
| 8 | 0.88 | 0.64 | 0.0093 | 0.0039 |

The quantum kernel values between different events get smaller as the number of qubits
grows, so the kernel matrix gets close to the identity matrix.

The classical SVM parameters (C and gamma) are chosen by 5-fold cross-validation on the
training set. The quantum SVM uses C = 1.

The paper does no tuning at all, C = 1 everywhere. With cross-validation the classical
AUC changes only in the third decimal, so the rounded table is the same for both settings.

## The ROOT reader

The numbers in the paper were produced with PyROOT. This package reads with uproot
instead, so that `pip install -e .` works without a ROOT build. Next step is to make the
reader a backend you can switch, with a second implementation over ROOT/RDataFrame, and
check that both give the same X and y on the same file.

The noisy simulation and the run on IBM hardware are in the paper but not yet in this
package.

## TODO

- noisy simulation with a device noise model
- run on IBM quantum hardware
- ROOT/RDataFrame reader next to the uproot one
- a switch for the paper settings (C = 1, no cross-validation)
- tests and CI