import regex as re
from typing import List, Tuple, Dict

# Standard BPE tokenization pattern for GPT-2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizerTrainer:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        # parameters
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # derived parameters
        self.init_vocab_size = 256 + len(special_tokens)
        self.num_merges = vocab_size - self.init_vocab_size
        
        # results
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def _load_corpus(self, input_path: str) -> str:
        """
        Load the text corpus from the specified file path.
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _init_vocab(self):
        """
        Initialize the vocabulary with byte values and special tokens.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode('utf-8')

    def pre_tokenize_parallel(self, corpus: str) -> Dict[Tuple[bytes, ...], int]:
        pass

    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        pass


def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the provided text corpus.

    This function trains a Byte-Pair Encoding (BPE) tokenizer by:
    1. Initializing vocabulary with 256 byte values and special tokens
    2. Pre-tokenizing the corpus using regex patterns
    3. Iteratively merging the most frequent byte pairs until vocab_size is reached

    Args:
        input_path: Path to a text file containing the BPE tokenizer training data.
            The file should contain UTF-8 encoded text.
        vocab_size: A positive integer defining the maximum final vocabulary size.
            This includes the initial 256 byte vocabulary, vocabulary items produced
            from merging, and any special tokens. Must be >= 256 + len(special_tokens).
        special_tokens: A list of strings to add to the vocabulary as special tokens.
            These tokens (e.g., '<|endoftext|>') will be assigned fixed token IDs and
            do not participate in the BPE merging process.

    Returns:
        A tuple containing:
            - vocab (Dict[int, bytes]): The tokenizer vocabulary, mapping from integer
              token ID to bytes representing the token. Token IDs are assigned sequentially
              starting from 0.
            - merges (List[Tuple[bytes, bytes]]): A list of BPE merges produced during
              training. Each element is a tuple (token1, token2) indicating that token1
              was merged with token2. The list is ordered by merge creation order, with
              earlier merges having higher priority during encoding.

    Example:
        >>> vocab, merges = bpe_train(
        ...     input_path="data/train.txt",
        ...     vocab_size=10000,
        ...     special_tokens=["<|endoftext|>"]
        ... )
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Number of merges: {len(merges)}")

    Note:
        - The function uses the GPT-2 regex pattern for pre-tokenization
        - Merges do not cross pre-token boundaries or special token boundaries
        - Ties in merge frequency are broken lexicographically (higher pair wins)
    """
    trainer = BPETokenizerTrainer(vocab_size, special_tokens)
    return trainer.train(input_path)