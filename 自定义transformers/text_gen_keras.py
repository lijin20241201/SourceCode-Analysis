import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import keras_nlp
import keras
import tensorflow as tf
import time
from keras import ops
from keras import layers
import numpy as np
def encode(text):
  text=text.numpy().decode('utf-8')
  # print(text)
  text_split=[]
  for i in str(text):
    if i.strip():
        text_split.append(i)
  idxs=[word2idx.get(i,1) for i in text_split]+[3]
  x=idxs[:-1]
  y=idxs[1:]
  return x,y
def tf_encode(text):
  # [tf.int64]返回几个值,这里就是几个值
  inp,tar = tf.py_function(encode, [text], [tf.int64,tf.int64]) 
  inp.set_shape([None])
  tar.set_shape([None])
  return inp,tar
def filter_max_length(x,y,max_length=80):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)
class TextGenerator(keras.callbacks.Callback): # keras 回调
    """从训练的模型生成文本.
    1. 给模型一些输入提示符
    2. 预测下一个token的概率
    3. 读取token，并将其添加到下一个输入中。
    """
    def __init__(
        self, max_tokens, start_tokens, index_to_word,word2idx, top_k=5, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.word2idx=word2idx
        self.print_every = print_every
        self.k = top_k
    def sample_from(self, logits):
        # 获取当前token中置信度最高的k个
        logits, indices = tf.math.top_k(logits, k=10, sorted=True)
        indices = indices.numpy()[0].astype("int64")  # 确保 indices 是 numpy 数组且为一维  
        preds = tf.nn.softmax(logits).numpy()[0].astype("float32")
        predicted_id = np.random.choice(indices, p=preds)  
        predicted_id = tf.constant(predicted_id, dtype=tf.int32)  
        predicted_id = tf.expand_dims([predicted_id], 0)  # 如果需要
        # 当 p 被指定时，它应该是一个与 a 中元素数量相同的一维数组或类似数组（如列表），并且数组
        # 中的每个元素都应该是一个非负的概率值，这些概率值的总和应该等于 1（或非常接近 1，以处理
        #浮点数的精度问题）。这样，numpy.random.choice（或类似的函数）就可以根据这些概率来随机选
        # 择 a 中的元素。
        return predicted_id
    def detokenize(self, number):
        return self.index_to_word[number]
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        text_split=[]
        for i in self.start_tokens:
            if i.strip():
                text_split.append(i)
        decoder_input = [self.word2idx.get(i,1) for i in text_split]
        output = tf.expand_dims(decoder_input, 0)
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            y = self.model.predict(output, verbose=0) # 获取模型预测
            # 采样模型预测的第i+1个token
            sample_token = self.sample_from(y[:,-1,:])
            if sample_token==3: # 如果遇到[END],就退出生成
                break
            tokens_generated.append(sample_token) # 把当前预测token加入模型生成token列表
            output = tf.concat([output, sample_token], axis=-1)
            num_tokens_generated = len(tokens_generated) # 统计生成token数
            
        txt = " ".join(
            [self.detokenize(_) for _ in tf.squeeze(output, axis=0).numpy().tolist()]
        )
        print(f"\n generated text:{txt}\n")
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    在自注意力（self-attention）机制中，通过将点积矩阵（dot product matrix）的上半部分进行掩码
    处理，可以阻止未来标记（tokens）的信息流向当前标记。这意味着，在点积矩阵中，从右下角开始计数的下
    三角部分填充为1（或保持不掩码，具体取决于实现方式），而上三角部分则通过掩码处理（通常设置为负无穷
    大或非常大的负数，以确保在softmax操作后这部分的权重接近于0），从而有效地阻断了信息的逆向流动。
    简单来说，这种掩码操作确保了模型在处理序列中的某个标记时，只能利用到该标记及其之前的所有信息，而不
    能“看到”或利用到序列中后续标记的信息。这对于处理序列数据（如文本或时间序列）中的因果关系至关重要，
    尤其是在自然语言处理（NLP）任务中，因为语言的生成和理解通常是顺序进行的，当前词或句子的理解依赖于
    之前的内容。
    """
    i = ops.arange(n_dest)[:, None] #(5,1)
    j = ops.arange(n_src) #(5)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src]) # 类似下三角的掩码(1,5,5)
    mult = ops.concatenate( #(8,1,1)
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult) # 为批次内所有样本复制因果掩码(8,5,5)
class TransformerBlock(layers.Layer): 
    # transformer解码器块,文本生成没有编码器输出部分,所以没有交叉注意力
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # 多头注意力
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential( # 前馈全连接层
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        # 层标准化
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate) # dropout
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs,mask=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0] # 批次大小
        seq_len = input_shape[1] # 序列长度
        #(8,5,5),下三角型掩码,因果掩码
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "int32")
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            causal_mask = ops.minimum(padding_mask, causal_mask)
        # query=key=value,自注意力,因果掩码，注意力输出,token加权和
        causal_mask=tf.cast(causal_mask,dtype='bool')
        attention_output = self.att(inputs, inputs, attention_mask= causal_mask)
        attention_output = self.dropout1(attention_output) 
        out1 = self.layernorm1(inputs + attention_output) # 自注意力前后残差
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output) # 前馈全连接前后残差
# 创建两个独立的嵌入层:一个用于token嵌入，另一个用于token在序列中的位置嵌入
class TokenAndPositionEmbedding(layers.Layer): 
    # 这种位置嵌入挺不错的,能让模型学习到位置信息,因为和词汇一样,都把位置，token的语义
    #转换成了向量,能够被训练,比写死的方式好多了
    def __init__(self, maxlen, vocab_size, embed_dim):
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
def bulid_model(vocab_size,embed_dim = 256,num_heads =4 ,feed_forward_dim = 512,layer_num=4):
    inputs = layers.Input(shape=(None,), dtype="int32")
     # 使用 Lambda 层来生成掩码  
    src_mask = layers.Lambda(lambda x:tf.not_equal(x, 0))(inputs) 
    # 这个位置可学习的，不能设置太大，设置小点参数少
    embedding_layer = TokenAndPositionEmbedding(120, vocab_size, embed_dim) # 嵌入层
    x = embedding_layer(inputs)
     # transformer解码器块,没有交叉注意力,src_mask
    for _ in range(layer_num):
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        x = transformer_block(x,src_mask)
    outputs = layers.Dense(vocab_size)(x) # 输出词汇表大小的置信度分布
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
class Poem_Loss(tf.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
        
    def call(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, 0) # 填充掩码
        loss_ = self.loss_fn(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype) # 数据类型转换
        loss_ *= mask # 这样填充位置的损失会是0,因为是0乘于那个数
        return tf.reduce_mean(loss_) 
        

    
sampler_dict={
    'GreedySampler':keras_nlp.samplers.GreedySampler(),
    'BeamSampler':keras_nlp.samplers.BeamSampler(num_beams=10),
    'RandomSampler':keras_nlp.samplers.RandomSampler(),
    'Top5Sampler':keras_nlp.samplers.TopKSampler(k=5),
    'Top10Sampler':keras_nlp.samplers.TopKSampler(k=10),
    'TopPSampler':keras_nlp.samplers.TopPSampler(p=0.5)
}
def get_prompt_idx(word2idx,prompt,seq_len):
    prom_lst=[]
    for i in prompt:
        if i.strip():
            prom_lst.append(i)
    prompt_tokens = [word2idx.get(_, 1) for _ in prom_lst]
    if len(prompt_tokens)<seq_len:
        prompt_tokens=prompt_tokens+[0]*(seq_len-len(prompt_tokens))
    elif len(prompt_tokens)>seq_len:
        raise ValueError(f'提示不能超过{seq_len}!')
    prompt_tokens=tf.convert_to_tensor([prompt_tokens],dtype='int32')
    index=len(prom_lst)
    return prompt_tokens,index
def get_generate_text(model,idx2word,prompt,index,sampler='GreedySampler'):
    def next(prompt, cache, index):
        logits = model(prompt,training=False)[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache
    output_tokens = sampler_dict[sampler](
    next=next,
    prompt=prompt,
    index=index,
    stop_token_ids=[3] #在模型预测到[END]时停止生成
    )
    txt =[idx2word.get(i,'[UNK]') for i in output_tokens.numpy()[0]]
    txt=''.join(txt).replace('[START]','').replace('[END]','').replace('[PAD]','')
    print(f"{sampler}生成的文本:{txt}")
def generate_poem(prompt,model,word2idx,idx2word,seq_len=80):
    prompt_tokens,index=get_prompt_idx(word2idx,prompt,seq_len)
    get_generate_text(model,idx2word,prompt_tokens,index)
    get_generate_text(model,idx2word,prompt_tokens,index,'BeamSampler')
    get_generate_text(model,idx2word,prompt_tokens,index,'RandomSampler')
    get_generate_text(model,idx2word,prompt_tokens,index,'Top5Sampler')
    get_generate_text(model,idx2word,prompt_tokens,index,'Top10Sampler')
    get_generate_text(model,idx2word,prompt_tokens,index,'TopPSampler')
