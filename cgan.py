# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Activation,BatchNormalization,Concatenate,Dense,
                                     Embedding,Flatten,Input,Multiply,Reshape)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam

img_rows=28
img_clos=28
channels=1

img_shape=(img_rows,img_clos,channels) 
z_dim=100
num_classes = 10

def print_progress(cur, fin):
    msg = '\rprogress: {0}/{1}'.format(cur, fin)
    print(msg, end='')

def build_generator(z_dim):
    model=Sequential()

    model.add(Dense(256*7*7,input_dim=z_dim))
    model.add(Reshape((7,7,256)))

    model.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same')) #(14,14,128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'))  #(14,14,64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))   #(28,28,1)
    model.add(Activation('tanh'))

    return model

def build_cgan_generator(z_dim):
    z=Input(shape=(z_dim,)) # 랜덤 잡음 벡터 z
    label=Input(shape=(1,),dtype='int32') # 조건 레이블 정수 0~9까지 생성자가 만들 숫자

    # 레이블 임베딩: 레이블을 z_dim 크기 밀집 벡터로 변환하고(batch_size,1,z_dim) 크기 3D 텐서를 만든다.
    # 밀집 벡터: 벡터의 표현법을 바꿔 더 작은 차원으로 표현하는 것
    # ex) 자연어 처리의 sparse/dense 표현의 예시
    # 강아지의 sparse 표현 = [ 0 0 0 0 1 0 0 0 0 0 0 0... 중략... 0]
    # 강아지의 dense 표현 = [0.2 1.8 1.1 -2.1 1.1 2.8... 중략...] # 이 벡터의 차원은 128
    label_embedding=Embedding(num_classes,z_dim,input_length=1)(label)
    # 임베딩된 3D 텐서를 펼쳐서 (batch_size,z_dim) 크기 2D 텐서로 바꾼다
    label_embedding=Flatten()(label_embedding)
    # 원소별 곱셈으로 잡음 벡터 z와 레이블 임베딩값을 결합
    joined_representation=Multiply()([z,label_embedding])

    generator=build_generator(z_dim)
    conditioned_img=generator(joined_representation)  # 주어진 레이블에 대한 이미지 생성

    # Model(input, output)
    return Model([z,label],conditioned_img)

def build_discriminator(img_shape):
    model=Sequential()
    
    model.add( # (28,28,2) 에서 (14,14,64) 텐서로 바꾸는 합성곱 층
              Conv2D(64,kernel_size=3,strides=2,input_shape=(img_shape[0],img_shape[1],img_shape[2] +1),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add( # (14,14,64) 에서 (7,7,64) 텐서로 바꾸는 합성곱 층
              Conv2D(64,kernel_size=3,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add( # (7,7,64) 에서 (3,3,128) 텐서로 바꾸는 합성곱 층
              Conv2D(128,kernel_size=3,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    return model

def build_cgan_discriminator(img_shape):
    img=Input(shape=img_shape)

    label=Input(shape=(1,),dtype='int32') # 입력 이미지의 레이블
    
    # 레이블 임베딩: 레이블을 z_dim 크기의 밀집 벡터로 변환하고 (batch_size,1,28x28x1) 크기의 3D 텐서를 만든다.
    label_embedding=Embedding(num_classes, np.prod(img_shape),input_length=1)(label)
    # 임베딩된 3D 텐서를 펼쳐서 (batch_size,28x28x1) 크기의 2D 텐서를 만든다
    label_embedding=Flatten()(label_embedding)
    # 레이블 임베딩 크기를 입력 이미지 차원과 동일하게 만든다
    label_embedding=Reshape(img_shape)(label_embedding)
    # 이미지와 레이블 임베딩을 연결한다
    concatenated=Concatenate(axis=-1)([img,label_embedding])

    discriminator=build_discriminator(img_shape)
    classification=discriminator(concatenated)  # 이미지-레이블 쌍을 분류한다
    
    model = Model([img,label],classification)
    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.00001),
                      metrics=['accuracy'])

    return model

def build_cgan(generator,discriminator):
    z=Input(shape=(z_dim,))  # 랜덤 잡음 벡터 z
    label=Input(shape=(1,))  # 이미지 레이블
    img=generator([z,label]) # 레이블에 맞는 이미지 생성하기

    classification=discriminator([img,label])

    model=Model([z,label],classification)
    model.compile(loss='binary_crossentropy',optimizer=Adam())
    return model

def sample_images(image_grid_rows=2, image_grid_columns=5, iter=0):

    # 랜덤한 잡음을 샘플링합니다.
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # 0–9 사이의 이미지 레이블을 만듭니다.
    labels = np.arange(0, 10).reshape(-1, 1)

    # 랜덤한 잡음에서 이미지를 생성합니다.
    gen_imgs = generator.predict([z, labels])

    # 이미지 픽셀 값을 [0, 1] 사이로 스케일을 변환합니다.
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 이미지 그리드를 설정합니다.
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(10, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 이미지 그리드를 출력합니다.
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1
    # plt.show()
    sample='results/cgan/sample {0}.pdf'.format(iter)
    plt.savefig(sample,dpi=150)

def train(iterations, batch_size, sample_interval):

    # MNIST 데이터셋을 로드합니다.
    (X_train, y_train), (_, _) = mnist.load_data()

    # [0, 255] 사이 흑백 픽셀 값을 [–1, 1]로 스케일 변환합니다.
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # 진짜 이미지의 레이블: 모두 1
    real = np.ones((batch_size, 1))

    # 가짜 이미지의 레이블: 모두 0
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  판별자를 훈련합니다.
        # -------------------------

        # 진짜 이미지와 레이블로 이루어진 랜덤한 배치를 얻습니다.
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # 가짜 이미지 배치를 생성합니다.
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict([z, labels])

        # 판별자를 훈련합니다.
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  생성자를 훈련합니다.
        # ---------------------

        # 잡음 벡터의 배치를 생성합니다.
        z = np.random.normal(0, 1, (batch_size, z_dim))

        # 랜덤한 레이블의 배치를 얻습니다.
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        # 생성자를 훈련합니다.
        g_loss = cgan.train_on_batch([z, labels], real)
        
        if (iteration + 1) % sample_interval == 0:

            # 훈련 과정을 출력합니다.
            print("\n%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))

            # 훈련이 끝난 후 그래프를 그리기 위해 손실과 정확도를 저장합니다.
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # 생성한 이미지 샘플을 출력합니다.
            sample_images(iter=iteration + 1)
        else:
            print_progress(iteration + 1, iterations)

        

discriminator=build_cgan_discriminator(img_shape)
discriminator.summary()

generator=build_cgan_generator(z_dim)
generator.summary()
discriminator.trainable=False

cgan=build_cgan(generator,discriminator)
cgan.summary()

accuracies = []
losses = []

iterations=25000
batch_size=32
sample_interval=iterations/10

train(iterations,batch_size,sample_interval)

# 그리드 차원을 설정합니다.
image_grid_rows = 10
image_grid_columns = 5

# 랜덤한 잡음을 샘플링합니다.
z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

# 생성할 이미지 레이블을 5개씩 준비합니다.
labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
labels_to_generate = labels_to_generate.flatten().reshape(-1, 1)

# 랜덤한 잡음에서 이미지를 생성합니다.
gen_imgs = generator.predict([z, labels_to_generate])

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

str1='results/cgan/result.pdf'
plt.savefig(str1,dpi=150)