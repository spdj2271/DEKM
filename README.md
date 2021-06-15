# DeepKmeans: Deep Embedded K-means Clustering
DeepKmeans

## Usage

1) Clone the code to local.
```
git clone https://github.com/spdj2271/DeepKmeans.git DeepKmeans
```
2) Launch an experiment on MNIST dataset.

```
cd DeepKmeans
python DeepKmeans.py
```

3)  Launch an experiment on other dataset, e.g., 'USPS', 'COIL20', 'FRGC'.
```
python DeepKmeans.py USPS
```



## Result
average results of three runs:
|            | MNIST  |        | USPS   |        | COIL-20 |        | FRGC   |        |
| ---------- | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ |
| Method     | ACC    | NMI    | ACC    | NMI    | ACC     | NMI    | ACC    | NMI    |
| DeepKmeans | 95.75 | 91.06 | 79.75 | 82.23 | 69.03  | 80.06 | 38.59 | 50.78 |

## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1
