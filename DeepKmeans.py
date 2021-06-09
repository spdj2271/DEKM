import csv
import os

import h5py
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling


def get_ACC_NMI(y, y_pred):
    s = np.unique(y_pred)
    t = np.unique(y)

    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(C)
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)
    acc = np.round(1.0 * count / len(y), 5)

    temp = np.array(y_pred)
    for i in range(N):
        y_pred[temp == col[i]] = i
    from sklearn.metrics import normalized_mutual_info_score
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    return acc, nmi


def get_xy(ds_name='MNIST', x_flatten=False, is_print=True):
    if ds_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)
    elif ds_name == 'USPS':
        with h5py.File('USPS.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        x = np.concatenate([X_tr, X_te], 0)
        y = np.concatenate([y_tr, y_te], 0)
        x = np.reshape(x, (len(x), 16, 16)).astype(np.float32)
    elif ds_name == 'COIL20':
        f = h5py.File('COIL20.h5', 'r')
        x = np.array(f['data'][()]).squeeze()
        x = np.expand_dims(np.swapaxes(x, 1, 2).astype(np.float32), -1)
        x = x / 255.
        y = np.array(f['labels'][()]).astype(np.float32)
        y[y == 20.] = 0.
        index = np.random.permutation(len(x))
        x = x[index]
        y = y[index]
        x = tf.image.resize(x, [28, 28]).numpy()
    elif ds_name == 'FRGC':
        with h5py.File('FRGC.h5', 'r') as hf:
            data = hf.get('data')[:]
            data = np.swapaxes(data, 1, 3)
            x = data / 255.
            y = hf.get('labels')[:]
            y_unique = np.unique(y)
            for i in range(len(y_unique)):
                y[y == y_unique[i]] = i
            index = np.random.permutation(len(x))
            x = x[index]
            y = y[index]
    if x_flatten:
        x = x.reshape((x.shape[0], -1))
    if is_print:
        print(ds_name)
    return x, y


def get_dataset_xx(ds_name='MNIST', x_flatten=False, is_print=True):
    x, _ = get_xy(ds_name=ds_name, x_flatten=x_flatten, is_print=is_print)
    ds = tf.data.Dataset.from_tensor_slices((x, x))
    return ds


def model_conv(load_weights=True):
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # init = 'uniform'
    filters = [32, 64, 128, hidden_units]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        input)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        x)
    x = layers.Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed')(x)
    # x=layers.BatchNormalization()(x)
#     x = tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1))
    h = x
    x = layers.Dense(filters[2] * (input_shape[0] // 8) * (input_shape[0] // 8), activation='relu')(x)
    x = layers.Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2]))(x)
    x = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding=pad3, activation='relu')(x)
    x = layers.Conv2DTranspose(filters[0], kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(input_shape[2], kernel_size=5, strides=2, padding='same')(x)
    output = layers.Concatenate()([h,
                                   layers.Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    if load_weights:
        model.load_weights('exp.h5')
        print('model_conv: weights was loaded')
    return model


def train_base():
    def loss_train_base(y_true, y_pred):
        y_true = layers.Flatten()(y_true)
        y_pred = y_pred[:, hidden_units:]
        return losses.mse(y_true, y_pred)

    model = model_conv(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    # model.compile(optimizer=tf.keras.optimizers.SGD(0.1,0.9), loss=loss_train_base)
    ds = get_dataset_xx(ds_name=ds_name).shuffle(10000).batch(batch_size_pretrain)
    model.fit(ds, epochs=epoch_pretrain, verbose=0)
    model.save_weights('exp.h5')


def show_reconstruct():
    x, y = get_xy(ds_name=ds_name)
    import matplotlib.pyplot as plt
    i = 1
    model = model_conv()
    for _ in range(10):
        plt.subplot(5, 4, i)
        seed = np.random.randint(0, 1000)
        image = np.expand_dims(x[seed], 0)
        plt.title(str(y[seed]))
        y_pred = model(image).numpy()[:, hidden_units:]
        y_pred = np.squeeze(y_pred)
        plt.axis('off')
        plt.imshow(np.squeeze(image).reshape(input_shape))
        plt.subplot(5, 4, i + 1)
        plt.axis('off')
        y_pred = np.reshape(y_pred, input_shape)
        # print(y_pred.shape)
        plt.imshow(np.squeeze(y_pred))
        i = i + 2
    plt.show()


def sorted_eig(X):
    e_vals, e_vecs = np.linalg.eig(X)  # 特征向量v[:,i]对应特征值w[i]，即每一列每一个特征向量
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def loss_finetuning(y_true, y_pred):
    V = tf.reshape(y_true[0, hidden_units:], (hidden_units, hidden_units))
    y_true = y_true[:, :hidden_units]
    y_pred = tf.matmul(y_pred[:, :hidden_units], V)
    return losses.mse(y_true, y_pred)


def train_ds():
    # Log_init
    dir_log = time.strftime('log/time_%m_%d__%H_%M' + '_ds_' + ds_name + '/')
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
    headers = ['Iter', 'ACC', 'NMI', 'Change_assignment', 'Time']
    f_log = open(dir_log + 'log_train.csv', 'w+', newline='')
    f_log_csv = csv.writer(f_log)
    f_log_csv.writerow(headers)
    f_log.flush()

    x, y = get_xy(ds_name=ds_name)
    assignment = np.array([-1] * len(x))
    model = model_conv()
    model.compile(optimizer='adam', loss=loss_finetuning)
    for iter in range(epoch_finetuning):
        # H=f(X)
        H = model.predict(x)[:, :hidden_units]
        kmeans_H = KMeans(n_clusters=n_clusters, n_init=500, max_iter=15000).fit(H)
        U = kmeans_H.cluster_centers_
        assignment_new = kmeans_H.labels_
        # 计算前后两轮U型能力，assignment变化了的样本个数
        w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for i in range(len(assignment_new)):
            w[assignment_new[i], assignment[i]] += 1
        ind = linear_assignment(-w)
        temp = np.array(assignment)
        for i in range(n_clusters):
            assignment[temp == ind[1][i]] = i
        change_assignment = np.sum(assignment_new != assignment)
        assignment = assignment_new

        acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))
        time_running = np.round(time.time() - time_start, 2)
        print('ite %2d ,acc: %.5f, nmi: %.5f' % (iter, acc, nmi),
              'change_assignment: ', change_assignment, 'time:', time_running)

        if change_assignment <= len(x) * 0.001:
            print('end')
            model.save_weights('final.h5')
            break

        # log
        ## 记录运行时间
        row = [iter, acc, nmi, change_assignment, time_running]
        f_log_csv.writerow(row)
        f_log.flush()
        ## 绘制embedding
        # tsne = TSNE(n_components=2)
        # x_2 = tsne.fit_transform(H, y)
        # colormap = cm.hsv(np.linspace(0, 1, 10))
        # plt.figure(0)
        # plt.scatter(x_2[:, 0], x_2[:, 1], c=colormap[np.array(y, dtype=np.int)], s=1, marker='x')
        # plt.axis('off')
        # plt.savefig(dir_log + 'iter_' + str(iter) + '_time_' + str(time_running) + '.pdf', dpi=500, bbox_inches='tight')

        S_i = []
        for i in range(n_clusters):
            temp = H[assignment == i] - U[i]
            temp = np.matmul(np.transpose(temp), temp)
            S_i.append(temp)
        S_i = np.array(S_i)
        S = np.sum(S_i, 0)
        Evals, V = sorted_eig(S)
        H_vt = np.matmul(H, V)  # 1000,5
        U_vt = np.matmul(U, V)  # 10,5

        # y_true = np.array(H_vt)
        # for i in range(len(y_true)):
        #     y_true[i, -1] = U_vt[assignment[i], -1]

        y_true = H_vt[:]
        temp = assignment[:]
        for i in range(len(y_true)):
            y_true[i, -1] = U_vt[temp[i], -1]

        V_y = np.reshape(V, (1, -1))
        V_y = np.tile(V_y, (len(y_true), 1))
        y_true = np.concatenate([y_true, V_y], -1)

        # fine-tuning
        ds = tf.data.Dataset.from_tensor_slices((x, y_true)).shuffle(10000).batch(batch_size_finetuning)
        model.fit(ds, epochs=epoch_T, verbose=0)
        # for x_batch, y_batch in ds.take(iteration_T):
        #     model.train_on_batch(x_batch, y_batch)
    model.save_weights('final.h5')


if __name__ == '__main__':
    epoch_pretrain = 200
    epoch_finetuning = 100
    batch_size_pretrain = 256
    batch_size_finetuning = 256
    # iteration_T = 40
    ds_name = 'MNIST'
    if ds_name == 'MNIST':
        batch_size_pretrain = 512
        batch_size_finetuning = 512
        epoch_T = 1
        input_shape = (28, 28, 1)
        n_clusters = 10
        hidden_units = n_clusters
    elif ds_name == 'USPS':
        epoch_T = 2
        input_shape = (16, 16, 1)
        n_clusters = 10
        hidden_units = n_clusters
    elif ds_name == 'COIL20':
        epoch_T = 10
        input_shape = (28, 28, 1)
        n_clusters = 20
        hidden_units = n_clusters
    elif ds_name == 'FRGC':
        epoch_T = 10
        input_shape = (32, 32, 3)
        n_clusters = 20
        hidden_units = n_clusters
    import time

    time_start = time.time()
    train_base()
    show_reconstruct()
    train_ds()
    print(time.time() - time_start)
