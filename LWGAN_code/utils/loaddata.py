import tensorflow as tf
import numpy as np


def get_syn_data(num_data=2000):
    sample = np.random.uniform(low=0.25, high=1, size=num_data)
    x = sample * np.cos(4 * np.pi * sample)
    y = sample * np.sin(4 * np.pi * sample)

    return np.vstack((x, y)).T


def get_data(dataset_name, data_shape, buffer_size, batch_size):
    if dataset_name == 'mnist':
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'cifar10':
        (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == 'cifar100':
        (train_images, train_labels), (_, _) = tf.keras.datasets.cifar100.load_data()
    elif dataset_name == 'fashion_mnist':
        (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'Synthetic2d':
        train_images = get_syn_data()
    else:
        raise ValueError('No datasets supported')

    train_images = train_images.reshape(train_images.shape[0], *data_shape).astype('float32')
    if dataset_name != 'Synthetic2d':
      train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = buffer_size
    BATCH_SIZE = batch_size

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    if dataset_name != 'Synthetic2d':
      ind = np.arange(0, train_images.shape[0])
      ind = np.random.choice(ind, size=10000, replace=False)
      test_data = train_images[ind, :]
      test_data = test_data.reshape(10000, -1)
    else:
      test_data = get_syn_data(10000)
      test_data = test_data.astype('float32')
      test_data = test_data.reshape(10000, -1)


    return train_dataset, test_data

# print(get_syn_data(10))
