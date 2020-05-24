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
# Train on 60000 samples
# Epoch 1/5
# 60000/60000 [==============================] - 4s 72us/sample - loss: 0.2919 - accuracy: 0.9156
# Epoch 2/5
# 60000/60000 [==============================] - 4s 58us/sample - loss: 0.1439 - accuracy: 0.9568
# Epoch 3/5
# 60000/60000 [==============================] - 4s 58us/sample - loss: 0.1080 - accuracy: 0.9671
# Epoch 4/5
# 60000/60000 [==============================] - 4s 59us/sample - loss: 0.0875 - accuracy: 0.9731
# Epoch 5/5
# 60000/60000 [==============================] - 3s 58us/sample - loss: 0.0744 - accuracy: 0.9766
# 10000/1 - 1s - loss: 0.0383 - accuracy: 0.9765