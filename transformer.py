import torch
from torch import Tensor
from torch.nn import Module, Embedding

# Custom local imports
from tokenizer import Tokenizer


class Transformer(Module):
    """
    Reimplementation of vanilla transformer for studying purposes.
    """

    def __init__(self, seq_len: int, dim: int, n_heads: int, depth: int, tokenizer: Tokenizer):
        self.seq_len: int = seq_len
        self.dim: int = dim
        self.n_heads: int = n_heads
        if (head_dim := (dim % n_heads)) != 0:
            raise RuntimeError(
                f"The dimension of the transformer needs to be divisible by the number of heads, however: 
                {dim=} % {n_heads=} == {dim % n_heads}"
            )
        self.head_dim: int = head_dim
        self.depth: int = depth
        # The transformer consists of an encoder and a decoder.
        # The encoder is trained for each token to self-attend to the whole sequence,
        self.encoder: Transformer.Encoder = self.Encoder()
        # The decoder is trained to self-attend only to previous tokens of the sequence
        # and cross-attend to the encoder's output.
        self.decoder: Transformer.Decoder = self.Decoder()
        self.tokenizer: Tokenizer = tokenizer
        # We need an embedding layer that learns useful representations of each token.
        self.token_embedding: Embedding = Embedding(num_embeddings = self.tokenizer.num_tokens, embedding_dim=dim)
        # Additionally, we need a positional embedding that makes it so the transformer isn't invariant to position.
        # This means, changing the embedding of a word based on its position in the sentence.
        # There are two choices here: Use a static embedding like sin-based as used in the original paper
        # or a trainable one.


    class Encoder(Module):
        pass

    class Decoder(Module):
        pass
    
    def sinusoidal_embedding(self, version: str = 'ML') -> Tensor:
        """
        Using a slightly different variant of the embedding, because my friend Mahan and I found it more beautiful.
        For PE(pos, 2i+1), we use cos(pos/10000^((2i+1)/d_model)) instead of cos(pos/10000^(2i/d_model)).
        We find the matching (2i+1) nicer.
        """
        # TODO: Test difference between standard version and ML (Mahan & Leon) version.
        s, d = self.seq_len, self.dim
        # 10000 is the proposed constant from the paper.
        n = 10000
        # Create tensor to hold the positional encodings for each position in sequence and dimension.
        embeddings = torch.zeros(s, d)

        positions = torch.arange(0, s).unsqueeze(1)

        match version:
            case 'ML':
                # For our own, very beautiful method, we need different denominators for the even and
                # positions in the embedding dimension.
                denominators_even = torch.pow(n, 2*torch.arange(0, d, 2) / d)
                demoninators_odd = torch.pow(n, 2*torch.arange(1, d, 2) / d)

                # Assign the actual sin and cos values to the positions in the embedding tensor.
                embeddings[:, 0::2] = torch.sin(positions/denominators_even)
                embeddings[:, 1::2] = torch.cos(positions/demoninators_odd)
            case _:
                # Standard implementation.
                denominators = torch.pow(n, 2*torch.arange(0, d//2) / d)

                # Assign actual values as above.
                embeddings[:, 0::2] = torch.sin(positions/denominators)
                embeddings[:, 1::2] = torch.cos(positions/denominators)

        return embeddings




def test():
    pass


if __name__ == "__main__":
    test()
