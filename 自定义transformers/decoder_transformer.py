import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import keras_nlp
import keras
import tensorflow as tf
import time
import numpy as np
from keras import ops
from keras import layers

# 创建两个独立的嵌入层:一个用于token嵌入，另一个用于token在序列中的位置嵌入
class TokenAndPositionEmbedding(layers.Layer): 
    def __init__(self,vocab_size, embed_dim,maxlen=512):
        super().__init__()
        #词嵌入,为词汇表中vocab_size的词元嵌入，每个token对应一个embed_dim维的嵌入向量
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #词在序列中的位置嵌入
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
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
    def call(self, q, k, v, mask):
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
    self.ffn = point_wise_feed_forward_network(dff,d_model)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
  def call(self, x,training,look_ahead_mask):
    attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask) 
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x) # 自注意力前后残差
    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    return out2, attn_weights_block1
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = TokenAndPositionEmbedding(target_vocab_size,d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x,training,look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # dropout只在训练模式时用
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x,training=training,look_ahead_mask=look_ahead_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,target_vocab_size,rate=0.1):
        super(Transformer, self).__init__()
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size,rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.dropout = tf.keras.layers.Dropout(0.3)
    def call(self,tar,training,look_ahead_mask):
        dec_output, attention_weights = self.decoder(
            tar,training=training,look_ahead_mask=look_ahead_mask)
        dec_output = self.dropout(dec_output,training=training)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)  # 输出层
        return final_output, attention_weights
