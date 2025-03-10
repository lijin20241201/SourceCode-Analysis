# 选择指标来衡量模型的损失和准确性。这些指标会累积各个时期的值，然后打印总体结果。
train_loss = tf.keras.metrics.Mean(name='train_loss')#损失
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')#准确率
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')


# @tf.function装饰器允许TensorFlow构建一个静态图以优化性能，但使用tf.GradientTape()
# 仍然是在动态图模式下进行的。这是因为tf.GradientTape()是用于在Eager Execution（Tenso
# rFlow的即时执行模式)中自动微分和计算梯度的。
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # 设置training=True的时候,模型内部应该有在训练和推理时行为不同的层(例如Dropout)
    # 如果模型内部没有dropout这种层,就没必要设置training=True
    predictions = model(images, training=True)
    # y_true,y_pre,sparse:会内部one-hot,如果已经one-hot,就得换
    # CategoricalCrossentropy损失
    loss = loss_object(labels, predictions) 
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_acc(labels, predictions)
# @tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  val_loss(t_loss)
  val_acc(labels, predictions)
def format_dataset(sl,xl):
    xl=tf.boolean_mask(xl, tf.not_equal(xl, 0)) 
    in_xl=xl[:-1]
    out_xl=xl[1:]
    if tf.shape(in_xl)[0]<seq_len:
        in_xl=tf.concat(\
            [in_xl, tf.zeros([seq_len - tf.shape(in_xl)[0]], dtype=in_xl.dtype)], axis=0)
    if tf.shape(out_xl)[0]<seq_len:
        out_xl = tf.concat(\
            [out_xl, tf.zeros([seq_len - tf.shape(out_xl)[0]], dtype=out_xl.dtype)], axis=0)
    return (
        {
            "encoder_inputs": sl,
            "decoder_inputs": in_xl,
        },
        out_xl,
    )