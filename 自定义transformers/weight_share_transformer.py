import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from keras import ops
import jieba
from pathlib import Path
import re
import keras_nlp
import random
# 根据测试,原序列加[END]效果更好,目标序列加[START]和[END],这样效果最好
def encode(sl,xl):
  sl,xl=sl.numpy().decode('utf-8'),xl.numpy().decode('utf-8')
  start_tensor=tf.constant([2], dtype=tf.int32)
  end_tensor = tf.constant([3], dtype=tf.int32)  
  sl=duilian_tokenizer.encode(sl)
  xl=duilian_tokenizer.encode(xl)
  sl,xl=tf.cast(sl,tf.int32),tf.cast(xl,tf.int32)
  sl = tf.concat([sl,end_tensor], axis=0) 
  xl = tf.concat([start_tensor,xl,end_tensor], axis=0) 
  return sl,xl
def tf_encode(sl,xl):
  result_sl, result_xl = tf.py_function(encode, [sl,xl], [tf.int32, tf.int32])
  result_sl.set_shape([None])
  result_xl.set_shape([None])
  return result_sl, result_xl
def filter_max_length(x, y, max_length=200):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

# 创建两个独立的嵌入层:一个用于token嵌入，另一个用于token在序列中的位置嵌入
class TokenAndPositionEmbedding(layers.Layer): 
    def __init__(self,vocab_size, embed_dim,maxlen=512):
        super().__init__()
        #词嵌入,为词汇表中vocab_size的词元嵌入，每个token对应一个embed_dim维的嵌入向量
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #词在序列中的位置嵌入
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def get_token_emb_weights(self):
        return self.token_emb.embeddings
    def call(self, x):
        maxlen = ops.shape(x)[-1] # 序列长度
        # 序列中每个token的位置索引
        positions = ops.arange(0, maxlen, 1) 
        # 位置嵌入,每个索引位置都会对应一个embed_dim大小的向量
        positions = self.pos_emb(positions) 
        x = self.token_emb(x)
        return x + positions # 返回带位置信息的token嵌入
def scaled_dot_product_attention(q, k, v, mask):
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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) #(n,s,h,dk)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (n,h,s,dk)
    def call(self,q,k,v,mask):
        batch_size = tf.shape(q)[0] # 批次大小
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, dk)
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
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(dff,d_model)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
      
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # 目标输入序列自注意力阶段,遮挡未来token,权重被分配给当前token之前的所有token
    attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask) 
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x) # 自注意力前后残差
    #mha2(v,k,q,...)
    #这里用的mask是原序列填充掩码,query是目标序列输入,在跨注意力中,每个目标token行对应的
    # 是原序列整个序列的token表示的加权,这里out1是目标序列自注意力输出,做query
    attn2, attn_weights_block2 = self.mha2( 
        out1,enc_output, enc_output, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,embed,
               rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = embed
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
      
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
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
    return x, attention_weights
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(dff,d_model)
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
class Encoder(tf.keras.layers.Layer): # 编码器
  def __init__(self, num_layers, d_model, num_heads, dff,embed,rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = embed
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
      
  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1] # 序列长度
    x = self.embedding(x) 
    x = self.dropout(x, training=training) # token嵌入表示
    
    for i in range(self.num_layers):
      # 按顺序用列表中的不同编码器层来encoder 
      x = self.enc_layers[i](x, training=training, mask=mask)
    return x  # (batch_size, input_seq_len, d_model)
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff,vocab_size,rate=0.1):
    super(Transformer, self).__init__()
    self.embed=TokenAndPositionEmbedding(vocab_size,d_model)
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           self.embed,rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           self.embed, rate)
    self.final_layer2=layers.Lambda(
        lambda x:tf.matmul(x,self.embed.get_token_emb_weights(), transpose_b=True))
    self.final_layer = tf.keras.layers.Dense(vocab_size)
    self.dropout = tf.keras.layers.Dropout(0.3)
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask,use_dense=True):
    enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)  
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training=training, 
        look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
    dec_output = self.dropout(dec_output,training=training)
    # (batch_size, tar_seq_len, target_vocab_size)
    if use_dense:
        final_output = self.final_layer(dec_output)
    else:
        # (n,s,d)@(d,v_size)-->(n,s,v_size)
        final_output = self.final_layer2(dec_output)
    return final_output, attention_weights
