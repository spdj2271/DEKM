import h5py
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


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


def get_xy(ds_type='MNIST', x_flatten=False, is_print=True):
    if ds_type == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)
    elif ds_type == 'USPS':
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
    elif ds_type == 'COIL20':
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
        # with h5py.File('COIL_2.h5', 'r') as hf:
        #     x = hf.get('X')[:]
        #     y = hf.get('Y')[:]
        #     x = x / 255.
        #     y[y == 20] = 0
        #     index = np.random.permutation(len(x))
        #     x = x[index]
        #     y = y[index]
        #     x = tf.image.resize(tf.expand_dims(x, -1), [28, 28]).numpy()
    elif ds_type == 'FRGC':
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
        print(ds_type)
    return x, y


def get_dataset_xx(ds_type='MNIST', x_flatten=False, is_print=True):
    x, _ = get_xy(ds_type=ds_type, x_flatten=x_flatten, is_print=is_print)
    ds = tf.data.Dataset.from_tensor_slices((x, x))
    return ds


def model_conv(load_weights=True):
    # init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    init = 'uniform'
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


def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)


def train_base():
    model = model_conv(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    # model.compile(optimizer=tf.keras.optimizers.SGD(0.1,0.9), loss=loss_train_base)
    ds = get_dataset_xx(ds_type=ds_name).shuffle(8000).batch(pretrain_batch_size)
    model.fit(ds, epochs=pretrain_epochs, verbose=2)
    # for i in range(pretrain_epochs):
    #     ds = get_dataset_xx(ds_type=ds_name).shuffle(8000).batch(pretrain_batch_size)
    #     for image, _ in ds:
    #         random = np.random.uniform()
    #         if random < 0.2:
    #             image = tf.image.random_contrast(image, 0.2, 0.5)
    #         elif random < 0.5:
    #             image = tf.image.random_brightness(image, 0.2)
    #         loss = model.train_on_batch(image, image)
    #     print(i, loss)
    model.save_weights('exp.h5')


def show_reconstruct():
    x, y = get_xy(ds_type=ds_name)
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


def train():
    model = model_conv()
    x, y = get_xy(ds_type=ds_name)
    optimizer = tf.keras.optimizers.Adam()
    update_interval = 40
    loss_value = 0
    index = 0
    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])
    for ite in range(int(140 * 100)):
        if ite % update_interval == 0:
            H = model(x).numpy()[:, :hidden_units]
            ans_kmeans = KMeans(n_clusters=n_clusters, n_init=500).fit(H)
            U = ans_kmeans.cluster_centers_
            assignment_new = ans_kmeans.labels_

            w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
            for i in range(len(assignment_new)):
                w[assignment_new[i], assignment[i]] += 1
            from scipy.optimize import linear_sum_assignment as linear_assignment
            ind = linear_assignment(-w)
            temp = np.array(assignment)
            for i in range(n_clusters):
                assignment[temp == ind[1][i]] = i
            change_assignment = np.sum(assignment_new != assignment)
            assignment = assignment_new

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
            #
            loss = np.round(np.mean(loss_value), 5)
            acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))
            print('ite %2d ,acc: %.5f, nmi: %.5f, loss:%.5f' % (ite // update_interval, acc, nmi, loss),
                  np.round(Evals[-5:], 2).astype(np.int), change_assignment,'time:',time.time()-time_start)
        if change_assignment<=len(x)*0.005:
            model.save_weights('final_DK.h5')
            print('end')
            break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        y_true = H_vt[idx]
        temp = assignment[idx]
        for i in range(len(idx)):
            y_true[i, -1] = U_vt[temp[i], -1]

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y_pred = model(x[idx])
            y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], V)
            loss_value = losses.mse(y_true, y_pred_cluster)+losses.mse(layers.Flatten()(x[idx]),y_pred[:,hidden_units:])
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0


if __name__ == '__main__':
    pretrain_epochs = 200
    pretrain_batch_size = 256
    batch_size = 256
    ds_name = 'MNIST'
    if ds_name == 'MNIST':
        input_shape = (28, 28, 1)
        n_clusters = 10
        hidden_units = n_clusters
    elif ds_name == 'USPS':
        input_shape = (16, 16, 1)
        n_clusters = 10
        hidden_units = n_clusters
    elif ds_name == 'COIL20':
        pretrain_batch_size = 32
        batch_size = 32
        input_shape = (28, 28, 1)
        n_clusters = 20
        hidden_units = n_clusters
    elif ds_name == 'FRGC':
        input_shape = (32, 32, 3)
        n_clusters = 20
        hidden_units = n_clusters
    import time
    time_start=time.time()
    train_base()
    show_reconstruct()
    train()
    print(time.time()-time_start)
