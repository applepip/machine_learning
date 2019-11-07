import tensorflow as tf

def BPnetwork(x_train, y_train):
    #使用层次模型
    model = tf.keras.Sequential()
    #输入层与第一个隐层同时建立，输入层10维，第一个隐层有10个神经元
    model.add(tf.keras.layers.Dense(10, input_dim=10))
    # 添加激活函数
    model.add(tf.keras.layers.Activation('relu'))
    #输出层
    model.add(tf.keras.layers.Dense(1, input_dim=10))
    # 添加激活函数
    model.add(tf.keras.layers.Activation('sigmoid'))
    #编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])
    #训练模型1000次
    model.fit(x_train, y_train, epochs = 100)
    # #保存模型权重
    # model.save_weights(modelfile)
    #返回训练模型
    return model