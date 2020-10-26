import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
# 对需要进行限制的GPU进行设置
# gpus[0] 默认 就是 第一张显卡
# memory_limit : 调整此处 更改model的能使用的显存上限, 实际申请显存会比这个数字高一点，不会高太多
tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2400)])

#更改此处调整  model模拟运行时 一次性处理的 图片数量， 太多会爆显存。
Batch_size = 25

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,GlobalAveragePooling2D

# from CRNN_test import getCrnnModel
# from Unet_test import getUnetModel

def getModels(classNum,imgHeight,imgWidth):
    model = Sequential([
        Conv2D(64, (3,3) ,padding='same', activation='relu', input_shape=(imgHeight, imgWidth ,3)),
        MaxPooling2D(),
        Conv2D(128,(3,3) ,padding='same',activation='relu'),
        MaxPooling2D(),
        Conv2D(256,(3,3) ,padding='same',activation='relu'),
        MaxPooling2D(),
        Conv2D(512,(3,3) ,padding='same',activation='relu'),
        MaxPooling2D(),
        Conv2D(1024,(3,3) ,padding='same',activation='relu'),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(512,activation='relu'),
        Dense(classNum,activation='softmax')
    ])
    # model.summary()
    return model

model = getModels(7,96,128)

# crnnModel = getCrnnModel()

# unetModel = getUnetModel(13,192,192)

while True:
    image = np.full((96,128,3),fill_value=1.25)
    image = tf.convert_to_tensor(image)
    image /= 255.0
    predictImg = tf.expand_dims(image,axis=0)

    image_batch = []
    for i in range(Batch_size):
        image_batch.append(predictImg)

    image_batch = tf.concat(image_batch,axis=0)

    prediction = model.predict(image_batch)

    """
    # unet------
    image = np.full((192,192,3),fill_value=3.25)
    image = tf.convert_to_tensor(image)
    image /= 255.0
    predictImg = tf.expand_dims(image,axis=0)

    image_batch = []
    for i in range(50):
        image_batch.append(predictImg)

    image_batch = tf.concat(image_batch,axis=0)

    prediction = unetModel.predict(image_batch)

    # crnn------

    image = np.full((32,100,3),fill_value=2.25)
    image = tf.convert_to_tensor(image)
    image /= 255.0
    predictImg = tf.expand_dims(image,axis=0)

    image_batch = []
    for i in range(50*12):
        image_batch.append(predictImg)

    image_batch = tf.concat(image_batch,axis=0)

    prediction = crnnModel.predict(image_batch)

    """