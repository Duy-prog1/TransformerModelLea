from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.nn import softmax
from numpy import random
 
# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, d_k, **kwargs):  # Initialize d_k here
        super(DotProductAttention, self).__init__(**kwargs)
        self.d_k = d_k
 
    def call(self, queries, keys, values, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
if __name__ == "__main__":
    print("Run test")
    input_seq_length = 5  # Maximum length of the input sequence
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    batch_size = 64  # Batch size from the training process
     
    queries = random.random((batch_size, input_seq_length, d_k))
    keys = random.random((batch_size, input_seq_length, d_k))
    values = random.random((batch_size, input_seq_length, d_v))
     
    attention = DotProductAttention(d_k=d_k) # Pass d_k during initialization
    print(attention(queries, keys, values)) # Call without d_k
