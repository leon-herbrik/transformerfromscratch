from typing import List, Union, Optional
import re


class Tokenizer:
    """
    A Tokenizer that creates tokens (integers representing (sequences of) characters) from a corpus.
    This is a very simple version just for studying that generates one token
    for each word and special character (including whitespaces and newlines).
    Modern versions of this use different and better tokenization operations.
    """

    # We use this regex to split the corpus into either a word (a continuous sequence of Ascii characters)
    # or a single character that is anything but a sequnce of Ascii characters (so anything but a 'word')
    r = r"([a-zA-Z]+|[^a-z^A-Z])"

    def __init__(
        self, corpus: Union[str, List[str]], special_tokens: Optional[List[str]] = None
    ) -> None:
        if isinstance(corpus, list):
            corpus = "\n".join(corpus)
        self.tokens = [i for i in special_tokens] if special_tokens is not None else []
        self.tokens += list(set(re.findall(self.r, corpus)))
        self.num_tokens = len(self.tokens)
        # Map from tokens to index of token.
        self.t_to_i = {token: i for i, token in enumerate(self.tokens)}
        # Map the other way.
        self.i_to_t = {i: token for i, token in enumerate(self.tokens)}

    def __getitem__(self, item: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        match item:
            case int():
                return self.i_to_t.get(item)
            case str():
                return self.t_to_i.get(item)
            case None:
                return None
            case _:
                raise NotImplementedError(
                    f"__getitem__ not implemented for data type: {type(item)}"
                )


def test():
    with open("corpus/sentences_cleansed.txt", "r") as f:
        lines = f.readlines()
    special_tokens = ["[MASK]", "<EOS>", "<SOS>"]
    tokenizer: Tokenizer = Tokenizer(lines, special_tokens=special_tokens)
    for s in special_tokens:
        print(tokenizer[s], tokenizer[tokenizer[s]])
    pass


def cleanse_words():
    with open("corpus/sentences.txt", "r") as f:
        lines = f.readlines()
        remove_first_part = r"^[0-9]+\s+"
        lines = [re.sub(remove_first_part, "", line) for line in lines]
    with open("corpus/sentences_cleansed.txt", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    test()
