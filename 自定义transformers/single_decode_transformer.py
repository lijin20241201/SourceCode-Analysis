import os
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  # 将 sin 应用于数组中的偶数索引（indices）；2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # 将 cos 应用于数组中的奇数索引；2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # 添加额外的维度来将填充加到
  # 注意力对数（logits）。
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask
def scaled_dot_product_attention(q, k, v, mask):
  """计算注意力权重。
  k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
  虽然 mask 根据其类型（填充或前瞻）有不同的形状，
  但是 mask 必须能进行广播转换以便求和。
  参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。
  返回值:
    输出，注意力权重
  """
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # 缩放 matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32) # dk
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # 注意力分数
  # 将 mask 加入到缩放的张量上。
  # 因为掩码中0表示非填充，1表示填充,mask * -1e9保证了填充是一个很大的负数
  # 而注意力分数和一个很大的负数想加也是一个很大的负数，而一个很大的负数
  # 的softmax输出是趋近于0,从而忽略了填充的加权值
  if mask is not None:
    scaled_attention_logits += (mask * -1e9) 
  # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  # (..., seq_len_q, seq_len_k)@(..., seq_len_v, depth_v)-->(..., seq_len_q, depth_v)
  # 因为 seq_len_k==seq_len_v
  output = tf.matmul(attention_weights, v)  
  return output, attention_weights
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads # 分成多个头,d_k
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)
  def split_heads(self, x, batch_size):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3]) #(n,h,s,d)
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0] # 批次大小
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    return output, attention_weights
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
      
  def call(self,inp,training,look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    # (batch_size, target_seq_len, d_model),因果掩码
    # 目标序列输入的前i个token,这里用的掩码是合并了因果和填充的掩码
    attn1, attn_weights_block1 = self.mha1(inp,inp,inp, mask=look_ahead_mask) 
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + inp) # 自注意力前后残差
    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    
    return out2, attn_weights_block1
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
  def call(self, x,training, look_ahead_mask):
    seq_len_x = tf.shape(x)[1]
    attention_weights = {}
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len_x, :]
    # dropout只在训练模式时用
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1 = self.dec_layers[i](x, training=training,
                                             look_ahead_mask=look_ahead_mask)
        
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,pe_target, rate=0.1):
    super(Transformer, self).__init__()
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
      
  def call(self, x,training,look_ahead_mask):
    
    dec_output, attention_weights = self.decoder(
        x,training=training, look_ahead_mask=look_ahead_mask )
    # (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.final_layer(dec_output)  # 输出层
    
    return final_output, attention_weights
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    step_float = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step_float) # 开方的倒数,步数越大，值越小
    arg2 = step_float * (self.warmup_steps ** -1.5) # 步数越大，值越大
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none') # 带 one-hot的,形状与label形状相同
def loss_function(real, pred):
  mask = tf.math.not_equal(real, 0) # 填充掩码
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype) # 数据类型转换
  loss_ *= mask # 这样填充位置的损失会是0,因为是0乘于那个数
  return tf.reduce_mean(loss_) # 聚合成平均token损失
def tf_encode(text):
  # [tf.int64]返回几个值,这里就是几个值
  inp,tar = tf.py_function(encode, [text], [tf.int64,tf.int64]) 
  inp.set_shape([None])
  tar.set_shape([None])
  return inp,tar
def create_masks(inp):#源序列,目标序列　
  look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
  # 目标序列填充掩码
  # inp_padding_mask = create_padding_mask(inp)
  # 合并的填充和因果掩码,maximum会返回对应位置的较大值,掩码中1表示填充,被遮挡
  # 或者在当前token后面,这个掩码是两者的结合,既遮挡了填充，又遮挡了当前token后面
  # combined_mask = tf.maximum(inp_padding_mask, look_ahead_mask)
  #返回编码器填充掩码，合并的掩码，解码器第二个注意力的源序列填充掩码
  return look_ahead_mask

def evaluate(promt):
  text_split=[]
  for i in promt:
    if i.strip():
        text_split.append(i)
  decoder_input = [word2idx.get(i,1) for i in text_split]
  output = tf.expand_dims(decoder_input, 0)
  for i in range(80):
    # print(output)
    # out是目标序列输入，每生成一个token,这个输入就多一个token,没有填充，可变长
    combined_mask=create_masks(output)
    # print(output)
    # predictions.shape == (batch_size, seq_len, vocab_size)
    # 随着目标序列输入的增加，这个预测中的seq_len会和目标序列输入长度一致
    predictions, attention_weights = transformer(output,
                                                 training=False,
                                                 look_ahead_mask=combined_mask)
    # 从 seq_len 维度选择最后一个词,这里选取最后一个token,因为前面的token都是之前
    # 模型已经预测过的,而这个token是这次模型预测的
    predictions = predictions[: ,-1, :]  # (batch_size, 1, vocab_size)
    # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    logits, indices = tf.math.top_k(predictions, k=5, sorted=True)
    indices = indices.numpy()[0].astype("int64")  # 确保 indices 是 numpy 数组且为一维  
    preds = tf.nn.softmax(logits).numpy()[0].astype("float32")
    predicted_id = np.random.choice(indices, p=preds)  
    predicted_id = tf.constant(predicted_id, dtype=tf.int32)  
    predicted_id = tf.expand_dims([predicted_id], 0)  # 如果需要
    # print(predicted_id) #模型预测的token对应目标词汇表中的索引
    # 如果 predicted_id 等于结束标记，就返回结果
    if predicted_id == 3: # 如果遇到[END]结束标记，立马返回输出
      return tf.squeeze(output, axis=0), attention_weights
    # 连接 predicted_id 与之前的目标序列输入，作为解码器的输入传递到解码器。
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights
def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)
  predicted_sentence = ''.join([idx2word.get(i,'UNK')for i in result.numpy().tolist()])
  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
