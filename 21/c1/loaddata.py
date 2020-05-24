# coding:utf-8
# 从tensorflow.examples.tutorials.mnist引入模块。这是TensorFlow为了教学MNIST而提前编制的程序
#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.keras as keras
# 从MNIST_data/中读取MNIST数据。这条语句在数据不存在时，会自动执行下载

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data("minst")

# 查看训练数据的大小
print(train_images.shape)  # (55000, 784)
print(len(train_labels))  # (55000, 10)

print(test_images.shape)  # (5000, 784)
print(len(test_labels))  # (5000, 10)

# 查看测试数据的大小
x_train, x_test = train_images / 255.0, test_images / 255.0

# 将模型的各层堆叠起来，以搭建 tf.keras.models.Sequential 模型，为训练选择优化器和损失函数
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练并验证模型
model.fit(x_train, train_labels, epochs=5)
model.evaluate(x_test, test_labels, verbose=2)
