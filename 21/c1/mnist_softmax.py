import tensorflow as tf
import tensorflow.keras as keras

# 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("minst")
x_train = x_train / 255
x_test = x_test / 255

# 配置模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 输入层
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 输出层，激活函数使用的是 softmax
])
# 配置交叉熵损失函数
loss = 'sparse_categorical_crossentropy'
# 配置 SGD，学习率为 0.1
optimizer = tf.keras.optimizers.SGD(0.1)
model.compile(optimizer=optimizer,
             loss = loss,
             metrics=['accuracy'])  # 使用准确率来评估模型

model.fit(x_train, y_train, epochs=5, batch_size=256)