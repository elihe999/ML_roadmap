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

def createConv2dLayer(core,size,x):
    x = tf.keras.layers.Conv2D(core,size,padding="same",activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(core,size,padding="same",activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def createUpConv2dLayer(core,size,x,conca):
    x = tf.keras.layers.Conv2DTranspose(core,size,strides=2,padding="same",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.concat([conca,x],axis=-1) #tf.contact
    return x


def getUnetModel(classNum,imgHeight,imgWidth):
    inputs = tf.keras.layers.Input(shape=(imgHeight,imgWidth,3))
    L1 = createConv2dLayer(64,3,inputs)
    L1_pool = tf.keras.layers.MaxPool2D(padding="same")(L1)

    L2 = createConv2dLayer(128,3,L1_pool)
    L2_pool = tf.keras.layers.MaxPool2D(padding="same")(L2)

    L3 = createConv2dLayer(256,3,L2_pool)
    L3_pool = tf.keras.layers.MaxPool2D(padding="same")(L3)

    L4 = createConv2dLayer(512,3,L3_pool)
    L4_pool = tf.keras.layers.MaxPool2D(padding="same")(L4)
    bottle = createConv2dLayer(1024,3,L4_pool)

    # # 上采样部位 右边
    R4 = createUpConv2dLayer(512,2,bottle,L4)
    R4 = createConv2dLayer(512,3,R4)

    R3 = createUpConv2dLayer(256,2,R4,L3)
    R3 = createConv2dLayer(256,3,R3)

    R2 = createUpConv2dLayer(128,2,R3,L2)
    R2 = createConv2dLayer(128,3,R2)

    R1 = createUpConv2dLayer(64,2,R2,L1)
    R1 = createConv2dLayer(64,3,R1)

    output = tf.keras.layers.Conv2D(classNum,1,padding="same",activation="softmax")(R1)

    return tf.keras.Model(inputs=inputs,outputs=output)



# 构造model
model = getUnetModel(13,192,192)

# 使用 numpy 填充的 tensor 来运行测试
image = np.full((192,192,3),fill_value=2.25)
image = tf.convert_to_tensor(image)
image /= 255.0
predictImg = tf.expand_dims(image,axis=0)

image_batch = []
for i in range(Batch_size):
    image_batch.append(predictImg)

image_batch = tf.concat(image_batch,axis=0)

while True:
    prediction = model.predict(image_batch)