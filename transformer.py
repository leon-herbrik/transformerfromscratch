from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, Embedding

# Custom local imports
from tokenizer import Tokenizer


class Transformer(Module):
    """
    Reimplementation of vanilla transformer for studying purposes.
    "https://arxiv.org/abs/1706.03762"
    """

    def __init__(
        self,
        seq_len: int = 2**10,
        dim: int = 768,
        n_heads: int = 8,
        depth: int = 8,
        tokenizer: Optional[Tokenizer] = None,
        positional_embedding: str = "ml",
    ):
        super().__init__()
        self.seq_len: int = seq_len
        self.dim: int = dim
        self.n_heads: int = n_heads

        self.depth: int = depth
        # The transformer consists of an encoder and a decoder.
        # The encoder is trained for each token to self-attend to the whole sequence.
        self.encoder: Transformer.Encoder = self.Encoder()
        # The decoder is trained to self-attend only to previous tokens of the sequence
        # and cross-attend to the encoder's output.
        self.decoder: Transformer.Decoder = self.Decoder()
        self.tokenizer: Tokenizer = (
            tokenizer
            if tokenizer is not None
            else Tokenizer(corpus=self.default_corpus())
        )
        # We need an embedding layer that learns useful representations of each token.
        self.token_embedding: Embedding = Embedding(
            num_embeddings=self.tokenizer.num_tokens, embedding_dim=dim
        )
        # Additionally, we need a positional embedding that makes it so the transformer isn't invariant to position.
        # This means, changing the embedding of a word based on its position in the sequence.
        # There are two choices here:
        #   - Use a static embedding like sin-based as in the original paper
        #   - or a trainable one.
        match positional_embedding:
            case "ml" | "standard":
                self.positional_embedding = self.Sinusoidal_Embedding(
                    self.dim, self.seq_len, version=positional_embedding
                )
            case "learned":
                self.positional_embedding = self.Learned_Embedding(
                    dim=self.dim, seq_len=self.seq_len
                )
            case _:
                raise NotImplementedError(
                    f'Positional embedding: {positional_embedding} is not implemented. Try "ml", "standard", or "learned".'
                )

    def forward(self, x):
        # Tokenize text.
        x = self.tokenizer(x, self.seq_len)
        x = Tensor(x).int()
        # Embed tokens.
        x = self.token_embedding(x)
        # Add positional embedding.
        x = self.positional_embedding(x)

        print(x.size())
        return x

    class Encoder(Module):
        """
        Encoder of the transformer. Consists of a stack of attention blocks that are specific to the encoder.
        """

        def __init__(
            self,
            seq_len: int = 2**10,
            dim: int = 768,
            n_heads: int = 8,
            depth: int = 8,
        ):
            self.seq_len: int = seq_len
            self.dim: int = dim
            self.n_heads: int = n_heads
            self.depth: int = depth
            self.head_dim, rest = divmod(dim, n_heads)
            if rest != 0:
                raise RuntimeError(
                    f"The embedding dimension needs to be divisible by the number of heads, however: {dim=} % {n_heads=} == {rest}"
                )

    class Decoder(Module):
        """
        Decoder of the transformer.
        """

        def __init__(
            self,
            seq_len: int = 2**10,
            dim: int = 768,
            n_heads: int = 8,
            depth: int = 8,
        ):
            self.seq_len: int = seq_len
            self.dim: int = dim
            self.n_heads: int = n_heads
            self.depth: int = depth
            self.head_dim, rest = divmod(dim, n_heads)
            if rest != 0:
                raise RuntimeError(
                    f"The embedding dimension needs to be divisible by the number of heads, however: {dim=} % {n_heads=} == {rest}"
                )

    class Learned_Embedding(Module):
        """
        This is a learned embedding with trainable parameters. Empirically there should be no difference.
        """

        def __init__(self, dim: int, seq_len: int):
            super().__init__()
            self.embeddings = Embedding(num_embeddings=seq_len, embedding_dim=dim)

        def forward(self, x):
            num_embeddings = self.embeddings.num_embeddings
            embedded = self.embeddings(torch.arange(num_embeddings))
            return x + embedded

    class Sinusoidal_Embedding(Module):
        """
        Using a slightly different variant of the embedding, because my friend Mahan and I found it more beautiful.
        For PE(pos, 2i+1), we use cos(pos/10000^((2i+1)/d_model)) instead of cos(pos/10000^(2i/d_model)).
        We find the matching (2i+1) nicer.
        """

        # TODO: Test difference between standard version and ML (Mahan & Leon) version.
        def __init__(self, dim: int, seq_len: int, version: str = "ml", n: int = 10000):
            super().__init__()
            # 10000 is the proposed constant from the paper.
            s, d = seq_len, dim
            # Create tensor to hold the positional encodings for each position in sequence and dimension.
            # This is a matrix holding values that can be added to the actual input data embeddings.
            self.embeddings = torch.zeros(s, d)

            positions = torch.arange(0, s).unsqueeze(1)

            # Select the correct version.
            match version:
                case "ml":
                    # For our own, very beautiful method, we need different denominators for the even and
                    # positions in the embedding dimension.
                    denominators_even = torch.pow(n, 2 * torch.arange(0, d, 2) / d)
                    demoninators_odd = torch.pow(n, 2 * torch.arange(1, d, 2) / d)

                    # Assign the actual sin and cos values to the positions in the embedding tensor.
                    self.embeddings[:, 0::2] = torch.sin(positions / denominators_even)
                    self.embeddings[:, 1::2] = torch.cos(positions / demoninators_odd)
                case "standard":
                    # Standard implementation.
                    denominators = torch.pow(n, 2 * torch.arange(0, d // 2) / d)

                    # Assign actual values as above.
                    self.embeddings[:, 0::2] = torch.sin(positions / denominators)
                    self.embeddings[:, 1::2] = torch.cos(positions / denominators)

        def forward(self, x):
            return x + self.embeddings

    def default_corpus(self) -> List[str]:
        """
        Some mock corpus for initializing the transformer if you don't have any corpus on hand.
        """
        corpus = [
            "The PC user is then able to resume surfing the Internet after he terminated the voice call\n",
            "You will have seen copies in the bar, and on the club noticeboard\n",
            "One of the crafts you will see operating on Coniston Water is the Steam Yacht Gondola operated by the National Trust\n"
            "Planes are extremely polluting and the fastest growing source of emissions\n"
            "Turn right at the next roundabout for entrance into the carpark.\n"
            "If there are accidents on the carpet, do not panic!\n",
        ]
        return corpus


def test():
    transformer = Transformer()
    print(transformer("The PC user is thon"))
    pass


if __name__ == "__main__":
    test()
