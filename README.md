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

3)  Launch an experiment other dataset, e.g., 'MNIST', 'USPS', 'COIL20', 'FRGC'.
```
python DeepKmeans.py USPS
```



## Result
Best result of five runs:
|            | MNIST  |        | USPS   |        | COIL-20 |        | FRGC   |        |
| ---------- | ------ | ------ | ------ | ------ | ------- | ------ | ------ | ------ |
| Method     | ACC    | NMI    | ACC    | NMI    | ACC     | NMI    | ACC    | NMI    |
| DeepKmeans | 96.744 | 92.446 | 79.297 | 80.921 | 75.069  | 82.898 | 38.911 | 51.143 |

## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1
