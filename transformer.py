from typing import List, Optional, Dict, Union

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Embedding,
    ModuleList,
    ModuleDict,
    Sequential,
    Linear,
    ReLU,
    LayerNorm,
)
import yaml
import dotsi

# Custom local imports
from tokenizer import Tokenizer


class Transformer(Module):
    """
    Reimplementation of vanilla transformer for studying purposes.
    "https://arxiv.org/abs/1706.03762"
    """

    def __init__(
        self,
        config: Optional[Union[Dict, dotsi.Dict]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        super().__init__()
        # Dotsi implements dot notation on dictionaries, making accessing the config variables a lot clearer imo.
        self.config = (
            dotsi.Dict(self.default_config()) if config is None else dotsi.Dict(config)
        )
        config = self.config
        config.n_heads, rest = divmod(config.dim, config.n_heads)
        if rest != 0:
            raise RuntimeError(
                f"The embedding dimension needs to be divisible by the number of heads, however: {config.dim=} % {config.n_heads=} == {rest}"
            )

        self.tokenizer: Tokenizer = (
            tokenizer
            if tokenizer is not None
            else Tokenizer(corpus=self.default_corpus())
        )
        # We need an embedding layer that learns useful representations of each token.
        self.token_embedding: Embedding = Embedding(
            num_embeddings=self.tokenizer.num_tokens, embedding_dim=config.dim
        )
        # Additionally, we need a positional embedding that makes it so the transformer isn't invariant to position.
        # This means, changing the embedding of a word based on its position in the sequence.
        # There are two choices here:
        #   - Use a static embedding like sin-based as in the original paper
        #   - or a trainable one.
        match config.positional_embedding:
            case "ml" | "standard":
                self.positional_embedding = self.Sinusoidal_Embedding(config)
            case "learned":
                self.positional_embedding = self.Learned_Embedding(config)
            case _:
                raise NotImplementedError(
                    f'Positional embedding: {config.positional_embedding} is not implemented. Try "ml", "standard", or "learned".'
                )
        # The transformer consists of an encoder and a decoder.
        # The encoder is trained for each token to self-attend to the whole sequence.
        self.encoder: Transformer.Encoder = self.Encoder(self)
        # The decoder is trained to self-attend only to previous tokens of the sequence
        # and cross-attend to the encoder's output.
        self.decoder: Transformer.Decoder = self.Decoder(self)

    def forward(self, x):
        # Tokenize text.
        x = self.tokenizer(x, self.config.seq_len)
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

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.blocks = Sequential(
                ModuleList([self.EncoderBlock(config) for i in range(config.depth)])
            )

        class EncoderBlock(Module):
            """
            One encoder block. Consists of bidirectional self-attention and feed-forward (1-hidden-layer-MLP) net.
            """

            def __init__(self, config):
                # TODO: Add layer norm.
                super().__init__()
                self.config = config
                self.attention = self.AttentionLayer(config)
                self.feed_forward = Sequential(
                    Linear(config.dim, config.dim),
                    ReLU(),
                    Linear(config.dim, config.dim),
                )

            def forward(self, x):
                x = self.attention(x)
                # Add & norm.
                x += x

                x = self.feed_forward(x)
                # Add & norm.
                x += x

            class AttentionLayer(Module):
                """
                Attention layer for use in Encoder stack.
                """

                def __init(self, config):
                    super().__init__()
                    self.config = config

                def forward(self, x):
                    # TODO: Implement.
                    return x

    class Decoder(Module):
        """
        Decoder of the transformer.
        """

        def __init__(self, transformer):
            super().__init__()

    class Learned_Embedding(Module):
        """
        This is a learned positional embedding with trainable parameters. Empirically there should be no difference
        between this and the sinusoidal one.
        """

        def __init__(self, config):
            super().__init__()
            self.embeddings = Embedding(
                num_embeddings=config.seq_len, embedding_dim=config.dim
            )

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
        def __init__(self, config):
            super().__init__()
            # 10000 is the proposed constant from the paper.
            s, d, n = config.seq_len, config.dim, config.sinusoidal_embedding_constant
            # Create tensor to hold the positional encodings for each position in sequence and dimension.
            # This is a matrix holding values that can be added to the actual input data embeddings.
            self.embeddings = torch.zeros(s, d)

            positions = torch.arange(0, s).unsqueeze(1)

            # Select the correct version.
            match config.version:
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

    def default_config(self) -> Dict:
        """
        Default config as yaml string, if no user defined config is given.
        """
        return yaml.safe_load(
            """
                seq_len: 1024,
                dim: 768,
                n_heads: 8,
                depth: 8,
                positional_embedding: 'ml'
                sinusoidal_embedding_constant: 10000
            """
        )


def test():
    transformer = Transformer()
    print(transformer("The PC user is thon"))
    pass


if __name__ == "__main__":
    test()
