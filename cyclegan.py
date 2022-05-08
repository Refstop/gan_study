import os
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE


class CYCLEGAN:
    def __init__(self):
        self.path = "results/cyclegan/"
        # Define the standard image size.
        self.orig_img_size = (286, 286)
        # Size of the random crops to be used during training.
        self.input_img_size = (256, 256, 3)

        self.buffer_size = 256
        self.batch_size = 1

        self.OUTPUT_CHANNELS = 3
        self.LAMBDA = 10
        self.EPOCHS = 40
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
            image, size=self.input_img_size)

        return cropped_image

    def normalize_img(self, img):
        img = tf.cast(img, dtype=tf.float32)
        # Map values in the range [-1, 1]
        return (img / 127.5) - 1.0

    def random_jitter(self, image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = self.random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image
    
    def preprocess_train_image(self, img, label):
        img = self.random_jitter(img)
        img = self.normalize_img(img)
        return img

    def preprocess_test_image(self, img, label):
        img = self.normalize_img(img)
        return img

    def load_hz_example(self):
        dataset, _ = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True)
        self.train_horses, self.train_zebras = dataset["trainA"], dataset["trainB"]
        self.test_horses, self.test_zebras = dataset["testA"], dataset["testB"]

        self.train_horses = (
            self.train_horses.map(self.preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
        )
        self.train_zebras = (
            self.train_zebras.map(self.preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
        )

        # Apply the preprocessing operations to the test data
        self.test_horses = (
            self.test_horses.map(self.preprocess_test_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
        )
        self.test_zebras = (
            self.test_zebras.map(self.preprocess_test_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
        )

        self.sample_horse = next(iter(self.train_horses))
        self.sample_zebra = next(iter(self.train_zebras))

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        # 다운샘플링은 컨벌루션을 사용
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
          result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        # 업샘플링은 컨벌루션 트랜스포즈(역컨벌루션과 비슷한 역할) 사용
        result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    # U-net 구조의 생성자 - pix2pix도 동일한 구조의 생성자를 사용한다.
    def build_generator(self, OUTPUT_CHANNELS):
        inputs = tf.keras.layers.Input(self.input_img_size)

        down_stack = [
          self.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
          self.downsample(128, 4), # (bs, 64, 64, 128)
          self.downsample(256, 4), # (bs, 32, 32, 256)
          self.downsample(512, 4), # (bs, 16, 16, 512)
          self.downsample(512, 4), # (bs, 8, 8, 512)
          self.downsample(512, 4), # (bs, 4, 4, 512)
          self.downsample(512, 4), # (bs, 2, 2, 512)
          self.downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
          self.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
          self.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
          self.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
          self.upsample(512, 4), # (bs, 16, 16, 1024)
          self.upsample(256, 4), # (bs, 32, 32, 512)
          self.upsample(128, 4), # (bs, 64, 64, 256)
          self.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
          x = down(x)
          skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
          x = up(x)
          x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def build_generator_gf(self):
        generator_g = self.build_generator(self.OUTPUT_CHANNELS)
        generator_f = self.build_generator(self.OUTPUT_CHANNELS)
        return generator_g, generator_f

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

        x = inp # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=inp, outputs=last)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def build_discriminator_xy(self):
        discriminator_x = self.build_discriminator()
        discriminator_y = self.build_discriminator()
        return discriminator_x, discriminator_y

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    # @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            self.generator_g, generator_f = self.build_generator_gf()
            discriminator_x, discriminator_y = self.build_discriminator_xy()

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

        generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Apply the gradients to the optimizer
        discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    def train(self, sample_interval):
        f = open(self.path + "progress.txt", 'a')
        for epoch in range(self.EPOCHS):
            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((self.train_horses, self.train_zebras)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print ("{} ".format(n), end='')
                n+=1

            clear_output(wait=True)
            # Using a consistent image (sample_horse) so that the progress of the model
            # is clearly visible.

            if (epoch + 1) % sample_interval == 0:
                # progress_info = "\n%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch + 1, d_loss[0], 100*d_loss[1], g_loss)
                # print(progress_info)
                # f.write(progress_info)
                # ckpt_save_path = ckpt_manager.save()
                # print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
                print('Saving sample for epoch {}'.format(epoch+1))
                self.generate_images(self.generator_g, self.sample_horse, epoch+1)
            else:
                self.print_progress(epoch + 1, self.EPOCH)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
        f.close()

    def generate_images(self, model, test_input, epoch=0):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            # plt.imshow(display_list[i] * 0.5 + 0.5)
            # plt.axis('off')
        plt.savefig(self.path + "{}.png".format(epoch))
        # plt.show()

    def print_progress(self, cur, fin):
        msg = '\rprogress: {0}/{1}'.format(cur, fin)
        print(msg, end='')

if __name__ == "__main__":
    cyclegan = CYCLEGAN()
    cyclegan.load_hz_example()
    cyclegan.train(sample_interval=10)
    cyclegan.generate_images(cyclegan.generator_g, cyclegan.sample_horse)
    # cyclegan.generator.save_weights(cyclegan.path + 'generator_weights.h5', overwrite=True)
    # cyclegan.discriminator.save_weights(cyclegan.path + 'discriminator_weights.h5', overwrite=True)