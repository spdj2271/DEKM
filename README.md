# Deep Embedded K-means Clustering
DEKM

## Usage

1) Clone the code to local.
```
git clone https://github.com/spdj2271/DEKM.git DEKM
```
2) Launch an experiment on MNIST dataset.

```
cd DEKM
python DEKM.py
```

3)  Launch an experiment on other dataset, e.g., 'USPS', 'COIL20', 'FRGC'.
```
python DEKM.py USPS
```

## Dataset
All datasets can be downloaded [here](https://drive.google.com/drive/folders/1raiYP1joy8gtsHXYcW5EuNtECSPRu37v?usp=sharing).

## Result
average results of three runs:
| Dataset    | ACC   | MNI   |
|------------|-------|-------|
| Conv       |       |       |
| MNIST      | 95.75 | 91.06 |
| USPS       | 79.75 | 82.23 |
| COIL-20    | 69.03 | 80.06 |
| FRGC       | 38.59 | 50.78 |
| Dense      |       |       |
| REUTES-10K | 76.28 | 59.06 |
| 20NEWS     | 41.08 | 40.28 |
| RCV1-10K   | 67.15 | 46.18 |

## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1
