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



## Result
average results of three runs:
|            | MNIST  |        | USPS   |        | COIL-20 |        | FRGC   |        |REUTES-10K   |        |20NEWS   |        |RCV1-10K   |        |
| ---------- | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ |------ | ------ |------ | ------ |------ | ------ |
| Method     | ACC    | NMI    | ACC    | NMI    | ACC     | NMI    | ACC    | NMI    | ACC    | NMI    | ACC    | NMI    | ACC    | NMI    |
| DEKM | 95.75 | 91.06 | 79.75 | 82.23 | 69.03  | 80.06 | 38.59 | 50.78 |38.59 | 50.78 |76.28| 59.06 |41.08 | 40.28 |67.15 | 46.18 |

## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1
