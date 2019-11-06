import tensorflow as tf

def BPnetwork(x_train, y_train, modelfile):
    #使用层次模型
    model = tf.keras.Sequential()
    #输入层与第一个隐层同时建立，输入层13维，第一个隐层有10个神经元
    model.add(tf.keras.Dense(10, input_dim=13))
    # 添加激活函数
    model.add(tf.keras.Activation('relu'))
    #输出层
    model.add(tf.keras.Dense(1,input_dim=12))
    #编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    #训练模型1000次
    model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 6)
    #保存模型权重
    model.save_weights(modelfile)
    #返回训练模型
    return model