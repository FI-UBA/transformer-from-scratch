import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Class representing the input embedding layer of a Transformer model.

    Args:
        d_model (int): The dimensionality of the embedding.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        d_model (int): The dimensionality of the embedding.
        vocab_size (int): The size of the vocabulary.
        embedding (nn.Embedding): The embedding lookup table.

    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # Embeddings dimension
        self.d_model = d_model
        # Vocabulary size -> How many words are in the vocabulary??
        self.vocab_size = vocab_size
        # Pytorch already provides a module for embeddings.
        # This module is a simple lookup table (learnable) that stores embeddings of a 
        # fixed dictionary and size.
        self.embedding = nn.Embedding(vocab_size, d_model) # (vocab_size, d_model)
        
    def forward(self, x):
        """
        Forward pass of the input embedding layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The embedded input tensor of shape (batch_size, sequence_length, d_model).

        """
        # According to the paper (section 3.4: Embeddings and Softmax), the embedding 
        # layer is multiplied by sqrt(d_model).
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int , seq_max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        # Embedding dimension
        self.d_model = d_model
        # Sequence Max length
        self.seq_max_len = seq_max_len
        # Dropout probability Layer
        self.dropout = nn.Dropout(p=dropout)
        #
        # See Topic 3.5 of paper Attention is all you need.
        # Create a positional encoding using a matrix of shape (seq_max_len, d_model)
        pe = torch.zeros(seq_max_len, d_model)
        # Create a vector of shape (seq_max_len, 1)
        position = torch.arange(0, seq_max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the denominator of the positional encoding function
        div_term = 10000**(torch.arange(0, d_model, 2).float() / d_model)
        # Apply the sin to the even indices of the sequence vector.
        # This means every word in the sequence but only even positions. 
        pe[:, 0::2] = torch.sin(position / div_term)
        # Apply the cos to the odd indices of the matrix
        pe[:, 1::2] = torch.cos(position / div_term)
        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)
        # Register the positional encoding as a buffer. 
        # This means that it will be saved in the model file.
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x = x + self.pe[:, :x.size(1)]
        # We do not need to learn the positional encoding, so we detach it from the computation graph.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Apply dropout and return the result (Section 5.4: Regularization)
        return self.dropout(x)
    
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        # super(LayerNormalization, self).__init__()
        super().__init__()
        # Save epsilon value used for numerical stability avoiding division by zero
        self.eps = eps
        # Add a learneable parameter called alpha to multiply the normalized value 
        self.alpha = nn.Parameter(torch.ones(1))
        # Add a learneable parameter called bias to add to the normalized value
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Calculate the mean of the input tensor
        mean = x.mean(dim=-1, keepdim=True) # Take the mean of the last dimension, so everything but the batch
        std = x.std(dim=-1, keepdim=True) # Take the mean of the last dimension, so everything but the batch
        # Normalize the input tensor
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm


# This class implement eq 2 in section 3.3 Position-wise Feed-Forward Networks of the paper
# Attention is all you need.
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        # super(FeedForwardBlock, self).__init__()
        super().__init__()
        # Fully connected layer 1
        self.fc1 = nn.Linear(d_model, d_ff)
        # Activation function
        self.activation = nn.ReLU()
        # Fully connected layer 2
        self.fc2 = nn.Linear(d_ff, d_model)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Here we get a tensor of shape (batch_size, seq_len, d_model)
        #
        # Apply the first fully connected layer
        x = self.fc1(x)
        # Apply the activation function
        x = self.activation(x)
        # Apply the second fully connected layer
        x = self.fc2(x)
        # Apply dropout
        x = self.dropout(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        # super(MultiHeadAttentionBlock, self).__init__()
        super().__init__()
        # Embedding dimension
        self.d_model = d_model
        # Number of heads
        self.num_heads = num_heads
        # Sanity check: d_model must be divisible by the number of heads
        assert d_model % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        #
        # Calculate the dimension of the head
        self.d_head = d_model // num_heads
        # Create the query, key and value linear layers used to learn matrices Wq, Wk and Wv
        self.wq = nn.Linear(d_model, d_model, bias=False)   # Wq
        self.wk = nn.Linear(d_model, d_model, bias=False)   # Wk
        self.wv = nn.Linear(d_model, d_model, bias=False)   # Wv
        # Create the output linear layer used to learn matrix Wo
        self.wo = nn.Linear(d_model, d_model, bias=False)   # Wo
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        #
        self.attention_scores = None

    @staticmethod
    def attention(query, key, value, mask = None, dropout: nn.Dropout = None):
        # Get the dimension of the head
        d_k = query.shape[-1]
        # Calculate the scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply the mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Replace the value of 0 with -1e9
        # Apply the softmax function
        p_attn = torch.nn.functional.softmax(scores, dim=-1) # (Batch, num_heads, Seq_Len, Seq_Len)
        # Apply dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        # Multiply the scores by the value matrix
        x = torch.matmul(p_attn, value)
        # Return the output and the attention scores for visualization
        return x, p_attn

    def forward(self, query, key, value, mask=None):
        # Perform the linear operation
        query = self.wq(query)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.wk(key)      # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.wv(value)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Split the query, key and value into different heads
        # Get the batch size
        batch_size = query.shape[0]
        # Split:
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, num_heads, d_head) -> (Batch, num_heads, Seq_Len, d_head)
        query = query.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Calculate the attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # d_model = num_heads * d_head

        # Apply the output linear layer
        x = self.wo(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        # super(ResidualConnection, self).__init__()
        super().__init__()
        # Add a layer normalization layer
        self.norm = LayerNormalization()
        # Add a dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Apply the sublayer
        x = x + self.dropout(sublayer(self.norm(x)))
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float = 0.1):
        # super(EncoderBlock, self).__init__()
        super().__init__()
        # Save the multi-head attention block
        self.self_attention_block = self_attention_block
        # Save the feed forward block
        self.feed_forward_block = feed_forward_block
        # Create the residual connections block. This can be done using nn.ModuleList() to create a list
        self.residual0 = ResidualConnection(dropout)
        self.residual1 = ResidualConnection(dropout)
        
    def forward(self, x, src_mask):
        # Apply the attention block
        x = self.residual0(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Apply the feed forward block
        x = self.residual1(x, self.feed_forward_block)
        return x


# As the encoder has N=6 times the EncoderBlock, we can create a class that 
# contains all the encoder blocks
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        # Save the encoder blocks
        self.layers = layers
        # Add a normalization layer
        self.norm = LayerNormalization()        
        
    def forward(self, x, src_mask):
        # Apply all the encoder blocks
        for layer in self.layers:
            x = layer(x, src_mask)
        # Apply the normalization layer and return
        return self.norm(x)

# class Encoder(nn.Module):
#     def __init__(self, layer: EncoderBlock, N: int):
#         # super(Encoder, self).__init__()
#         super().__init__()
#         # Create N encoder blocks
#         self.layers = nn.ModuleList([layer for _ in range(N)])
#         # Add a normalization layer
#         self.norm = LayerNormalization()        
        
#     def forward(self, x, src_mask):
#         for layer in self.layers:
#             x = layer(x, src_mask)
#         return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float = 0.1):
        # super(DecoderBlock, self).__init__()
        super().__init__()
        # Save the self attention block
        self.self_attention_block = self_attention_block
        # Save the cross attention block (came from las EncoderBlock)
        self.cross_attention_block = cross_attention_block
        # Save the feed forward block 
        self.feed_forward_block = feed_forward_block
        # Create the residual connections block. This time let's use nn.ModuleList() to create a list
        self.residual = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) 
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # NOTE: "encoder_output" is the output of the encoder, and we have src_mask (source mask) and tgt_mask (target mask)
        #       because we are programming the transformer for machine translation task!.

        # Apply the self attention block
        x = self.residual[0](x, lambda x: self.self_attention_block(query=x, key=x, value=x, mask=tgt_mask))
        # Apply the source attention block
        x = self.residual[1](x, lambda x: self.cross_attention_block(query=x, key=encoder_output, value=encoder_output, mask=src_mask))
        # Apply the feed forward block
        x = self.residual[2](x, self.feed_forward_block)
        return x


# As we did with the encoder, we can create a class that contains all the decoder blocks
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        # super(Encoder, self).__init__()
        super().__init__()
        # Save the decoder blocks
        self.layers = layers
        # Add a normalization layer
        self.norm = LayerNormalization()        
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply all the encoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply the normalization layer and return
        return self.norm(x)

# class Decoder(nn.Module):
#     def __init__(self, N: int = 6):
#         # super(Encoder, self).__init__()
#         super().__init__()
#         # Save the decoder blocks
#         self.layers = nn.ModuleList([DecoderBlock(.....) for _ in range(N)])
#         # Add a normalization layer
#         self.norm = LayerNormalization()        
        
#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         # Apply all the encoder blocks
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         # Apply the normalization layer and return
#         return self.norm(x)


# This class is the last in the model stack. It is used to project the output of the decoder to the 
# vocabulary size.
# NOTE: The output of the decoder is a tensor of shape (Batch, Seq_Len, d_model) and we need to project it 
# to the vocabulary size. We include the Softmax function HERE!.
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        # super(ProjectionLayer, self).__init__()
        super().__init__()
        # Save the linear layer
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, vocab_size)
        return torch.nn.functional.softmax(self.linear(x), dim=-1)


# Put all the pieces together
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, generator: ProjectionLayer):
        # super(Transformer, self).__init__()
        super().__init__()
        # Save the encoder
        self.encoder = encoder
        # Save the decoder
        self.decoder = decoder
        # Save the source embedding
        self.src_embed = src_embed
        # Save the target embedding
        self.tgt_embed = tgt_embed
        # Save the source positional encoding
        self.src_pos = src_pos
        # Save the target positional encoding
        self.tgt_pos = tgt_pos
        # Save the projection layer
        self.generator = generator

    # Method to encode the source sentence
    def encode(self, src, src_mask):
        # Generate the embeddings
        src = self.src_embed(src)
        # Apply the positional encoding
        src = self.src_pos(src)
        # Apply the encoder
        return self.encoder(src, src_mask)

    # Method to decode the target sentence
    def decode(self, encoder_output, tgt, src_mask, tgt_mask):
        # Generate the embeddings
        tgt = self.tgt_embed(tgt)
        # Apply the positional encoding
        tgt = self.tgt_pos(tgt)
        # Apply the decoder
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # Method to generate the output
    def project(self, x):
        return self.generator(x)
    
# class Transformer1(nn.Module):
#     def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, N: int = 6, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
#         # super(Transformer, self).__init__()
#         super().__init__()
#         # Create the input embedding layer
#         self.src_embed = InputEmbedding(d_model, src_vocab_size)
#         self.tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
#         # Create the positional encoding layer
#         self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
#         # Create the encoder
#         self.encoder = Encoder(nn.ModuleList([EncoderBlock(MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)]))
#         # Create the decoder
#         self.decoder = Decoder(nn.ModuleList([DecoderBlock(MultiHeadAttentionBlock(d_model, num_heads, dropout), MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)]))
#         # Create the projection layer
#         self.projection = ProjectionLayer(d_model, tgt_vocab_size)
        
#     def forward(self, src, tgt, src_mask, tgt_mask):
#         # Apply the input embedding layer
#         src = self.pos_enc(self.src_embed(src))
#         tgt = self.pos_enc(self.tgt_embed(tgt))
#         # Apply the encoder
#         encoder_output = self.encoder(src, src_mask)
#         # Apply the decoder
#         x = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
#         # Apply the projection layer
#         return self.projection(x)


# Method to build the model
def build_transformer_model(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, Nenc: int = 6, Ndec: int = 6, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
    # Create the input embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    # Create the positional encoding layers.
    # pos_enc = PositionalEncoding(d_model, dropout=dropout)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout=dropout)
    # Create the encoder
    encoder = Encoder(nn.ModuleList([EncoderBlock(MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(Nenc)]))
    # Create the decoder
    decoder = Decoder(nn.ModuleList([DecoderBlock(MultiHeadAttentionBlock(d_model, num_heads, dropout), MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(Ndec)]))
    # Create the projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    # Create the transformer model
    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    # Initialize the weights of the model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Return the model
    return model
