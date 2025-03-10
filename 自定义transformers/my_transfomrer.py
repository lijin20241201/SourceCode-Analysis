import os
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# 根据测试,原序列加[END]效果更好,目标序列加[START]和[END],这样效果最好
def encode(lang1, lang2):
  lang1,lang2=lang1.numpy().decode('utf-8'),lang2.numpy().decode('utf-8')
  start_tensor=tf.constant([2], dtype=tf.int32)
  end_tensor = tf.constant([3], dtype=tf.int32)  
  lang1=zh_tokenizer.encode(lang1)
  lang2=en_tokenizer.encode(lang2)
  lang1,lang2=tf.cast(lang1,tf.int32),tf.cast(lang2,tf.int32)
  lang1 = tf.concat([lang1,end_tensor], axis=0) 
  lang2 = tf.concat([start_tensor,lang2,end_tensor], axis=0) 
  return lang1, lang2
def tf_encode(zh,en):
  result_zh, result_en = tf.py_function(encode, [zh,en], [tf.int32, tf.int32])
  result_zh.set_shape([None])
  result_en.set_shape([None])
  return result_zh, result_en
def filter_max_length(x, y, max_length=100):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
# transformer用这种固定的位置嵌入没用可训练的位置嵌入效果好
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
  return mask  # (seq_len, seq_len)

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
  def call(self,q, k, v,mask):
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
# 点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。
def point_wise_feed_forward_network(dff,d_model):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(dff, d_model)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask=mask)  # (batch_size, input_seq_len, d_model)
    # 注意:dropout与训练状态有关
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    return out2
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(dff, d_model)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
      
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    # (batch_size, target_seq_len, d_model),因果掩码
    # 目标序列输入的前i个token,这里用的掩码是合并了因果和填充的掩码
    attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask) 
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x) # 自注意力前后残差
    #mha2(v,k,q,...)
    #这里用的mask是原序列掩码
    attn2, attn_weights_block2 = self.mha2( 
        out1, enc_output, enc_output, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2
class Encoder(tf.keras.layers.Layer): # 编码器
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    # 一个包含多个编码器层的对象,目的就是在内存里用不同的空间
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
      
  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1] # 序列长度
    # 将嵌入和位置编码相加。
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training) # token嵌入表示
    
    for i in range(self.num_layers):
      # 按顺序用列表中的不同编码器层来encoder 
      x = self.enc_layers[i](x, training=training, mask=mask)
    
    return x  # (batch_size, input_seq_len, d_model)
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
      
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    # dropout只在训练模式时用
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      # 传人参数:目标序列前i个token,编码器的输出
      #x的值只在第一次时是嵌入，后面都是decode的输出
      x, block1, block2 = self.dec_layers[i](x, enc_output, training=training,
                                             look_ahead_mask=look_ahead_mask, 
                                             padding_mask=padding_mask)
        
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    self.dropout = tf.keras.layers.Dropout(0.5)
      
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    # (batch_size, inp_seq_len, d_model)
    # 注意:这里传入的是源序列填充掩码
    enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)  
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # look_ahead_mask:因果掩码,dec_padding_mask:注意,这里传人的是
    # 目标序列填充掩码
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training=training, 
        look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
    dec_output = self.dropout(dec_output,training=training)
    # (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.final_layer(dec_output)  # 输出层
    return final_output, attention_weights
# 根据论文中的公式，将 Adam 优化器与自定义的学习速率调度程序（scheduler）配合使用。
# 自定义的学习率调节器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step_float) # 开方的倒数,步数越大，值越小
        arg2 = step_float * (self.warmup_steps ** -1.5) # 步数越大，值越大
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    def get_config(self):
         # 无需调用super().get_config()，因为基类可能不提供此方法 
        # config = super().get_config()
        return {  
            "d_model": float(self.d_model.numpy()),  # 将Tensor转换为Python float，以便在JSON中序列化  
            "warmup_steps": self.warmup_steps  
        }  
    @classmethod  
    def from_config(cls, config):  
        # 注意：这里需要确保config中的d_model是float，因为我们在get_config中将其转换为了float  
        return cls(d_model=config['d_model'], warmup_steps=config['warmup_steps']) 
  
# 由于目标序列是填充（padded）过的，因此在计算损失函数时，应用填充遮挡非常重要。
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none') # 带 one-hot的,形状与label形状相同

def loss_function(real, pred):
  mask = tf.math.not_equal(real, 0) # 填充掩码
  loss = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss.dtype) # 数据类型转换
  loss *= mask # 这样填充位置的损失会是0,因为是0乘于那个数
  return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def create_masks(inp, tar): # 源序列,目标序列　
  # 编码器原序列填充遮挡
  enc_padding_mask = create_padding_mask(inp)
  # 在解码器的第二个注意力模块使用。
  # 该填充遮挡用于遮挡编码器的输出。注意:这里是源序列
  dec_padding_mask = create_padding_mask(inp)
  # 在解码器的第一个注意力模块使用。
  # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
  #look_ahead_mask:因果掩码，里面1表示遮挡
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  # 目标序列填充掩码
  # dec_target_padding_mask = create_padding_mask(tar)
  # 合并的填充和因果掩码,maximum会返回对应位置的较大值,掩码中1表示填充,被遮挡
  # 或者在当前token后面,这个掩码是两者的结合,既遮挡了填充，又遮挡了当前token后面
  # combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  #返回编码器填充掩码，合并的掩码，解码器第二个注意力的源序列填充掩码
  return enc_padding_mask,look_ahead_mask, dec_padding_mask
  # return enc_padding_mask, combined_mask, dec_padding_mask

step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=step_signature)
def train_step(inp, tar):
  #目标序列输入包括start和end和一堆填充
  tar_inp = tar[:, :-1] 
  # 目标序列真实标签包括:对应目标序列输入的下一个token和end和一堆填充
  # 之所以目标序列输入有start,是可以从编码器输出里预测目标序列第一个真实token
  tar_real = tar[:, 1:]
  #编码器填充掩码，合并了因果掩码和填充掩码的掩码，在解码器第二个注意力要输入的原序列掩码
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  # (64, 1, 1, 22) (64, 1, 19, 19) (64, 1, 1, 22)
  # print(enc_padding_mask.shape,combined_mask.shape,dec_padding_mask.shape)
  with tf.GradientTape() as tape:
    # enc_padding_mask:编码器源序列掩码,dec_padding_mask:解码器跨注意力阶段源序列掩码
    # 用于遮挡编码器输出中的填充,combined_mask:解码器自注意力阶段掩码,包括合并的填充和
    # 因果掩码，用于遮挡目标输入序列中填充和当前token之后的token
    predictions, _ = transformer(inp, tar_inp, 
                                 training=True, 
                                 enc_padding_mask=enc_padding_mask, 
                                 look_ahead_mask=combined_mask, 
                                 dec_padding_mask=dec_padding_mask)
    # 计算真实下个token和预测的下个token的误差
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  train_loss(loss) # 更新loss指标
  train_acc(tar_real, predictions) # 更新acc指标

from keras import ops

def evaluate(inp_sentence):
  start_token = [2]
  end_token = [3]
  inp_sentence =zh_tokenizer.encode(inp_sentence)+end_token
  encoder_input = tf.convert_to_tensor([inp_sentence],dtype=tf.int32)
  decoder_input = [2]
  output = tf.expand_dims(decoder_input, 0)
  output=tf.cast(output,tf.int32)
  for i in range(MAX_SEQUENCE_LENGTH):
    # out是目标序列输入，每生成一个token,这个输入就多一个token,没有填充，可变长
    enc_padding_mask, combined_mask, dec_padding_mask = my_encoder_decoder_transformer.create_masks(
        encoder_input, output)
    # 随着目标序列输入的增加，这个预测中的seq_len会和目标序列输入长度一致
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 training=False,
                                                 enc_padding_mask=enc_padding_mask,
                                                 look_ahead_mask=combined_mask,
                                                 dec_padding_mask=dec_padding_mask)
    predictions = predictions[: ,-1:, :] 
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    if predicted_id == 3: # 如果遇到[END]结束标记，立马返回输出
      return tf.squeeze(output, axis=0), attention_weights
    # 连接 predicted_id 与之前的目标序列输入，作为解码器的输入传递到解码器。
    output = tf.concat([output,predicted_id], axis=-1)
    # print(output)
  return tf.squeeze(output, axis=0), attention_weights

def translate(test_sample, plot=''):
  input_sentence=test_sample[0]
  result, attention_weights = evaluate(input_sentence)
  result=result.numpy().tolist()
  predicted_sentence = en_tokenizer.decode(result) 
  predicted_sentence=(predicted_sentence.replace("[PAD]", "") 
        .replace("[START]", "") # 替换掉开始符和结束符
        .replace("[END]", "")
        .replace('@@ ',"").replace(' &apos;',"'").strip())
  src_sent=input_sentence.replace('@@','').replace(' ','').strip()
  tgt_sent=test_sample[1].replace(' &apos;',"'").replace('@@ ','').strip() 
  print(src_sent,'|',predicted_sentence,'|',tgt_sent)

def decode_sequence(input_sentence):
    src_idxs = zh_tokenizer.encode(input_sentence)+[eos_idx]
    src_idxs=tf.convert_to_tensor([src_idxs],dtype='int32')
    decoded_sentence='[START]'
    for i in range(MAX_SEQUENCE_LENGTH):
        tgt_idxs = en_tokenizer.encode(decoded_sentence)
        tgt_idxs=tf.convert_to_tensor([tgt_idxs],dtype='int32')
        predictions,_ = transformer(src_idxs,tgt_idxs,
                                             training=False,
                                             enc_padding_mask=None,
                                             look_ahead_mask=None,
                                             dec_padding_mask=None)
        # print(predictions.shape)
        sampled_token_index = ops.argmax(predictions[0,i, :]).numpy() # 当前预测的token索引
        # print(sampled_token_index)
        sampled_token = en_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[END]":
            break
    return decoded_sentence
    