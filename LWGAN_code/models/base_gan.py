import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from models.discriminator import get_bjorck_discriminator, get_clipped_discriminator
from models.generator import get_generator
from tqdm import tqdm

from utils.wasserstein_dist import wasserstein_dist


class BaseGAN(object):

    def __init__(self,
                 z_shape=50,
                 out_dim=(28, 28, 1),
                 epochs=50,
                 batchsize=256,
                 test_num=100,
                 g_depth=5,
                 g_width=64,
                 d_depth=5,
                 d_width=64,
                 lrg=1e-4,
                 lrd=1e-4,
                 beta_1=0.9,
                 beta_2=0.999,
                 bjorck_beta=0.5,
                 bjorck_iter=5,
                 bjorck_order=2,
                 group_size=2,
                 num_critic=5):
        self.z_shape = z_shape
        self.out_dim = out_dim
        self.epochs = epochs
        self.batchsize = batchsize
        self.test_num = test_num
        self.num_critic = num_critic

        self.d_depth = d_depth
        self.d_width = d_width
        self.g_depth = g_depth
        self.g_width = g_width

        # network initialization
        self.G = get_generator(input_shape=(self.z_shape,), output_shape=self.out_dim, depth=g_depth, width=g_width)

        self.D = get_bjorck_discriminator(input_shape=self.out_dim, depth=d_depth, width=d_width,
                                          bjorck_beta=bjorck_beta, bjorck_iter=bjorck_iter, bjorck_order=bjorck_order,
                                          group_size=group_size)

        # self.D = get_clipped_discriminator(input_shape=self.out_dim, depth=d_depth, width=d_width)
        self.G_optimizer = Adam(learning_rate=lrg, beta_1=beta_1, beta_2=beta_2)
        self.D_optimizer = Adam(learning_rate=lrd, beta_1=beta_1, beta_2=beta_2)
        # self.G_optimizer = RMSprop(learning_rate=lrg)
        # self.D_optimizer = RMSprop(learning_rate=lrd)
        self.Loss = 0.0

    def generator_loss(self, fake_output):
        return tf.math.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return -tf.math.reduce_mean(fake_output) + tf.math.reduce_mean(real_output)

    @tf.function
    def train_step_discriminator(self, images, batchsize):
        # noises = tf.random.uniform([batchsize, self.z_shape], minval=0, maxval=1, dtype=tf.dtypes.float32)
        noises = tf.random.normal([batchsize, self.z_shape])
        with tf.GradientTape() as disc_tape:
            generated_images = self.G(noises, training=False)

            real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=True)

            # gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        # self.G_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    @tf.function
    def train_step_generator(self, batchsize):
        # noises = tf.random.uniform([batchsize, self.z_shape], minval=0, maxval=1, dtype=tf.dtypes.float32)
        noises = tf.random.normal([batchsize, self.z_shape])
        with tf.GradientTape() as gen_tape:
            generated_images = self.G(noises, training=True)

            # real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=False)

            gen_loss = self.generator_loss(fake_output)
            # disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        # gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        # self.D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    def generate_sample(self, num):
        # noise = tf.random.uniform(shape=[num, self.z_shape], minval=0, maxval=1, dtype=tf.dtypes.float32)
        noise = tf.random.normal([num, self.z_shape])
        return self.G(noise, training=False)

    def generate_and_save_images(self, epochs, num, test_images):
        predictions = self.generate_sample(num)
        # print(predictions.shape)
        test = test_images[0:num, :]
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        plt.scatter(test[:, 0], test[:, 1], color=color1, label='real data')
        plt.scatter(predictions[:, 0, :], predictions[:, 1, :], color=color2, label='generated data')
        plt.legend()

        # plt.scatter(test_images[:, 0, :], test_images[:, 1, :])
        plt.savefig(
            'img/stat_normal_at_epoch_{:04d}_{}_{}_{}_{}.png'.format(epochs, self.g_depth, self.g_width, self.d_depth,
                                                                     self.d_width))

    def train(self, dataset_images, test_images):
        print('--------------Begin Training-----------------')
        count = 0
        for epoch in tqdm(range(self.epochs)):
            # start = time.time()

            for image_batch in dataset_images:
                self.train_step_discriminator(image_batch, self.batchsize)
                count = count + 1
                if (count % self.num_critic == 0):
                    self.train_step_generator(self.batchsize)
                    count = 0

            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        # tf.print(tf.reduce_max(tf.abs(self.D.get_layer(index=1).trainable_variables[0])))
        xs = self.generate_sample(num=10000)
        # xs = np.clip(xs, a_min=-1, a_max=1)
        xs = np.reshape(xs, newshape=(-1, np.prod(self.out_dim, dtype=int)))
        xt = test_images
        Loss = self.discriminator_loss(self.D(xs, training=False), self.D(xt, training=False))
        print('Loss is {}'.format(Loss))
        W = wasserstein_dist(xs, xt)
        print('Wasserstein Loss is {}:'.format(W))
        # np.save("6000_generated.npy", xs)


        # self.generate_and_save_images(self.epochs, self.test_num, test_images)
        print('--------------End Training-----------------')


'''
    @tf.function
    def train_step(self, images, batchsize):
        noises = tf.random.uniform([batchsize, self.z_shape],minval=0, maxval=1, dtype=tf.dtypes.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.G(noises, training=True)

            real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))
'''
