import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, multiply, Reshape, Concatenate
from tensorflow.keras.layers import BatchNormalization, Embedding, Flatten, Dropout, Activation
from tensorflow.keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.optimizers import Adam

class CGAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        self.build_cgan()


    def build_generator(self, mode):
        model = Sequential()
        if mode == "custom":
            model.add(Dense(256, input_dim = self.latent_dim))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(np.prod(self.img_shape)))
            ## 사고해볼것 - 구별자의 input에 맞추기 위해 아무 생각 없이 reshape 해버렸지만...이래도 되는 걸까?
            ## 결과: 학습 결과는 확인되지만, accuracy와 loss가 이상하다. 왜지?
            model.add(Reshape((28,28,1)))
            # model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))   #(28,28,1)
            model.add(Activation('tanh'))
        elif mode == "keras":
            model.add(Dense(256*7*7, input_dim=self.latent_dim))
            model.add(Reshape((7, 7, 256)))

            model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')) #(14,14,128)
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.01))

            model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))  #(14,14,64)
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.01))

            model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))   #(28,28,1)
            model.add(Activation('tanh'))
        # model.summary()

        return model

    def build_cgan_generator(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        ## 레이블 임베딩: 레이블을 z_dim 크기 밀집 벡터로 변환하고(num_classes,1,latent_dim) 크기 3D 텐서를 만든다.
        ## 밀집 벡터: 벡터의 표현법을 바꿔 더 작은 차원으로 표현하는 것
        ## ex) 자연어 처리의 sparse/dense 표현의 예시
        ## 강아지의 sparse 표현 = [ 0 0 0 0 1 0 0 0 0 0 0 0... 중략... 0], 이 벡터의 차원은 1000
        ## 강아지의 dense 표현 = [0.2 1.8 1.1 -2.1 1.1 2.8... 중략...], 이 벡터의 차원은 128
        # model = Sequential()
        ## Flatten <- Embedding <- label 순으로 쌓아올림
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        ## 단일 벡터의 앙상블(모델 결합) - 그냥 요소별 곱셈
        model_input = multiply([noise, label_embedding])

        img = self.build_generator("custom")(model_input)

        return Model([noise, label], img)


    def build_discriminator(self):
        model = Sequential()

        ## 28x28x1 -> 14*14*64
        ## 왜 28*28*2??
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(self.img_shape[0],self.img_shape[1],self.img_shape[2]+1)))
        model.add(LeakyReLU(alpha=0.01))
        ## 14*14*64 -> 7*7*64
        model.add(Dropout(0.4))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        ## 7*7*64 -> 3*3*128
        model.add(Dropout(0.4))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))

        # model.summary()

        return model


    def build_cgan_discriminator(self):
        img = Input(shape=self.img_shape)
        label = Input(shape=(1,))

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape), input_length=1)(label))

        ## 레이블 임베딩 크기를 입력 이미지 차원과 동일하게 만든다
        label_embedding=Reshape(self.img_shape)(label_embedding)
        ## 이미지와 레이블 임베딩을 연결한다
        ## Concatenate는 두 다른 종류의 모델을 결합하는 앙상블 방법의 일종으로
        ## 아래 코드에서는 img input 레이어와 label_embedding 레이어를 결합한다.
        model_input=Concatenate(axis=-1)([img,label_embedding])
        
        validity = self.build_discriminator()(model_input)

        return Model([img, label], validity)

    def build_cgan(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        self.discriminator = self.build_cgan_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        self.discriminator.summary()

        self.generator = self.build_cgan_generator()
        self.generator.summary()
        # input만 준다
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        # input만 준다
        valid = self.discriminator([img, label])

        self.cgan = Model([noise, label], valid)
        self.cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        ## -1~1 범위 내로 normalize
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        ## (60000, 28, 28) -> (60000, 28, 28, 1) 네트워크 인풋에 맞추기 위해서
        X_train = np.expand_dims(X_train, axis=3)
        ## reshape(-1, 1): -1은 남은 다른 쪽에 자동으로 크기를 맞춤을 의미.
        ## ex) 크기 (3,4) ->reshape(-1, 1)-> 크기 (12, 1)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        ## 진짜 이미지의 레이블: 전부 1
        valid = np.ones((batch_size, 1))
        ## 가짜 이미지의 레이블: 전부 0
        fake = np.zeros((batch_size, 1))
        f = open("results/cgan_custom/progress.txt", 'a')
        for epoch in range(epochs):
            ## 학습시킬 레이블을 랜덤으로 지정
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            ## 가짜 이미지를 만들 재료, 노이즈 생성
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            ## label에 대한 가짜 이미지 생성
            gen_imgs = self.generator.predict([noise, labels])

            ## 구별자가 진짜 이미지에 대해서 학습
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            ## 구별자가 가짜 이미지에 대해서 학습
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            ## 총 오차 계산
            d_loss = np.add(d_loss_real, d_loss_fake)

            ## 생성자를 학습시킬 레이블
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1,1)
            ## 진짜와 가짜 이미지에 대해서 학습한 구별자에 대한 결과를 생성자가 학습
            g_loss = self.cgan.train_on_batch([noise, sampled_labels], valid)

            if epoch % sample_interval == 0:
                progress_info = "\n%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)
                print(progress_info)
                f.write(progress_info)
                self.sample_images(epoch)
            else:
                self.print_progress(epoch + 1, epochs)
        f.close()

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
        fig.savefig("results/cgan_custom/%d.png" % epoch)
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

        str1='results/cgan_custom/result.pdf'
        plt.savefig(str1,dpi=150)

    def print_progress(self, cur, fin):
        msg = '\rprogress: {0}/{1}'.format(cur, fin)
        print(msg, end='')

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=25000, batch_size=32, sample_interval=1000)
    cgan.sample_test()
    cgan.generator.save_weights('results/cgan_custom/generator_weights.h5', overwrite=True)
    cgan.discriminator.save_weights('results/cgan_custom/discriminator_weights.h5', overwrite=True)