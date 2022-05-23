# Deep Embedded K-means Clustering
[Paper](https://arxiv.org/pdf/2109.15149.pdf)

## Usage

1) Clone the code locally.
```
git clone https://github.com/spdj2271/DEKM.git DEKM
```
2) Launch an experiment on MNIST dataset.

```
cd DEKM
python DEKM.py MNIST
```

3)  Launch an experiment on other dataset, e.g., 'USPS', 'COIL20', 'FRGC'. 
```
python DEKM.py USPS
```
- All datasets can be downloaded [here](https://drive.google.com/drive/folders/1raiYP1joy8gtsHXYcW5EuNtECSPRu37v?usp=sharing). 
- When launch experiments on other datasets (except 'MNIST'), you should make sure you have the following folder structure:
```
|-- undefined
    |-- DEKM.py
    |-- DEKM_dense.py
    |-- utils.py
    |-- datasets
    |   |-- 20NEWS
    |   |   |-- test_data.npz
    |   |   |-- test_label.npz
    |   |   |-- train_data.npz
    |   |   |-- train_label.npz
    |   |-- COIL20
    |   |   |-- COIL20.h5
    |   |-- FRGC
    |   |   |-- FRGC.h5
    |   |-- RCV1
    |   |   |-- test
    |   |   |-- validation
    |   |-- REUTERS
    |   |   |-- 10k_feature.npy
    |   |   |-- 10k_target.npy
    |   |-- USPS
    |       |-- USPS.h5
```

## Result
Average results of three runs:
| image (Conv) dataset | ACC   | MNI   | text (Dense) dataset | ACC   | MNI   |
|--------------|-------|-------|---------------|-------|-------|
| MNIST        | 95.75 | 91.06 | REUTES-10K    | 76.28 | 59.06 |
| USPS         | 79.75 | 82.23 | 20NEWS        | 41.08 | 40.28 |
| COIL-20      | 69.03 | 80.06 | RCV1-10K      | 67.15 | 46.18 |
| FRGC         | 38.59 | 50.78 |               |       |       |

## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1

## Citations
If you find our project helpful, your citations are highly appreciated:
```
@inproceedings{guo2021deep,
  title={Deep Embedded K-Means Clustering},
  author={Guo, Wengang and Lin, Kaiyan and Ye, Wei},
  booktitle={2021 International Conference on Data Mining Workshops (ICDMW)},
  pages={686--694},
  year={2021},
  organization={IEEE}
}
```
