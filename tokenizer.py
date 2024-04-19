from typing import List, Union, Optional
import re

from torch import Tensor


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
        self.tokens += ["<UKN>"]
        self.tokens += ["[MASK]"]
        self.tokens += self.extract_tokens(corpus)
        self.num_tokens = len(self.tokens)
        # Map from tokens to index of token.
        self.t_to_i = {token: i for i, token in enumerate(self.tokens)}
        # Map the other way.
        self.i_to_t = {i: token for i, token in enumerate(self.tokens)}

    def __getitem__(
        self, items: Union[List, str, int, None]
    ) -> List[Union[None, str, int]]:
        if not isinstance(items, list):
            items = [items]
        results = []
        for element in items:
            match element:
                case int():
                    # Return stored token or <UKN> if the id is not present.
                    results.append(self.i_to_t.get(element, "<UKN>"))
                case str():
                    # Return stored id or the id of <UKN> if the token is not present.
                    results.append(self.t_to_i.get(element, self.t_to_i.get("<UKN>")))
                case None:
                    results.append(None)
                case _:
                    raise NotImplementedError(
                        f"__getitem__ not implemented for data type: {type(element)}"
                    )
        if len(results) == 1:
            results = results[0]
        return results

    def __call__(self, x: str, sequence_length: int = 0):
        """
        @param sequence_length: Append [MASK] tokens until the list of tokens is as long as sequence_length
        """
        tokens = self.extract_tokens(x, remove_duplicates=False)
        ids = self[tokens]
        if sequence_length > len(ids):
            ids += [self.t_to_i["[MASK]"]] * (sequence_length - len(ids))
        return ids

    def extract_tokens(self, string: str, remove_duplicates=True):
        matches = re.findall(self.r, string)
        # Use this to preserve order.
        tokens = list(dict.fromkeys(matches)) if remove_duplicates else matches
        return tokens


def test():
    with open("corpus/sentences_cleansed.txt", "r") as f:
        lines = f.readlines()
    special_tokens = ["<EOS>", "<SOS>"]
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
