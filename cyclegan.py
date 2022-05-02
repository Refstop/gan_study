import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, multiply, Reshape, Concatenate
from tensorflow.keras.layers import BatchNormalization, Embedding, Flatten, Dropout, Activation
from tensorflow.keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

class CYCLEGAN:
    def __init__(self):
        self.path = "results/cyclegan/"
        pass

    def load_hz_example(self):
        pass

    def build_generator_xy(self):
        pass

    def build_generator_yx(self):
        pass

    def build_discriminator_x(self):
        pass

    def build_discriminator_y(self):
        pass

    def build_cyclegan(self):
        pass

    def train(self):
        pass

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.path + "{}.png".format(epoch))
        plt.close()

    def sample_test(self):
        # 그리드 차원을 설정합니다.
        image_grid_rows = 10
        image_grid_columns = 5

        # 랜덤한 잡음을 샘플링합니다.
        z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, self.latent_dim))

        # 생성할 이미지 레이블을 5개씩 준비합니다.
        labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
        labels_to_generate = labels_to_generate.flatten().reshape(-1, 1)

        # 랜덤한 잡음에서 이미지를 생성합니다.
        gen_imgs = self.generator.predict([z, labels_to_generate])

        # 이미지 픽셀 값을 [0, 1] 사이로 스케일을 변환합니다.
        gen_imgs = 0.5 * gen_imgs + 0.5

        # 이미지 그리드를 설정합니다.
        fig, axs = plt.subplots(image_grid_rows,
                                image_grid_columns,
                                figsize=(10, 20),
                                sharey=True,
                                sharex=True)

        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                # 이미지 그리드를 출력합니다.
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title("Digit: %d" % labels_to_generate[cnt])  ## NEW
                # str1='results/cgan/Digit: {0}.pdf'.format(labels_to_generate[cnt])
                # plt.savefig(str1,dpi=150)
                cnt += 1

        str1=self.path + "result.png"
        plt.savefig(str1,dpi=150)

    def print_progress(self, cur, fin):
        msg = '\rprogress: {0}/{1}'.format(cur, fin)
        print(msg, end='')

if __name__ == "__main__":
    cyclegan = CYCLEGAN()
    cyclegan.train(epochs=25000, batch_size=32, sample_interval=1000)
    cyclegan.sample_test()
    cyclegan.generator.save_weights(cyclegan.path + 'generator_weights.h5', overwrite=True)
    cyclegan.discriminator.save_weights(cyclegan.path + 'discriminator_weights.h5', overwrite=True)