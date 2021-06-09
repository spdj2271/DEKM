# DeepKmeans: Deep Embedded K-means Clustering
DeepKmeans

## Usage
Run python Deepkmeans.py  to run experiment. The dataset is specified by $ds_name$, e.g., 'MNIST', 'USPS', 'COIL20', 'FRGC'.

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
