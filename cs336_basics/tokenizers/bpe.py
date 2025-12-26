import regex as re
from typing import List, Tuple, Dict, BinaryIO
from multiprocessing import Pool
import mmap

# Standard BPE tokenization pattern for GPT-2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizerTrainer:
    def __init__(
        self,
        vocab_size: int, 
        special_tokens: List[str],
        num_workers: int = 4
    ):
        # parameters
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_workers = num_workers

        # derived parameters
        self.init_vocab_size = 256 + len(special_tokens)
        self.num_merges = vocab_size - self.init_vocab_size
        
        # results
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def _find_chunk_boundaries(self, file: BinaryIO) -> List[int]:
        """
        Find chunk boundaries in the file based on special tokens.
        Args:
            file: A binary file object opened in 'rb' mode.
        Returns:
            A list of byte offsets representing chunk boundaries.

        Note: This function is used to split the file for parallel processing.
        And it uses mmap for efficient file access.
        """
        # spilit the whole file into chunks based on special tokens
        mmaped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        boundaries = [0]

        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            pos = 0
            while (pos := mmaped.find(token_bytes, pos)) != -1:
                boundaries.append(pos)
                pos += len(token_bytes)

        boundaries.append(mmaped.size())
        mmaped.close()
        return sorted(set(boundaries))

    def _init_vocab(self):
        """
        Initialize the vocabulary with byte values and special tokens.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode('utf-8')

    def _pre_tokenize_parallel(self, boundaries: List[int]) -> Dict[Tuple[bytes, ...], int]:
        """"
        Pre-tokenize the text corpus in parallel using the defined regex pattern with multiprocessing.
        Args:
            boundaries: List of byte offsets representing chunk boundaries.
        Returns:
            A dictionary mapping pre-token tuples to their frequency counts.
        """
        # prepare chunks
        chunks = [(self.input_path, boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        def worker(chunk_info: Tuple[str, int, int]) -> Dict[Tuple[bytes, ...], int]:
            file_path, start, end = chunk_info
            # get the chunk bytes
            with open(file_path, 'rb') as f:
                f.seek(start)
                chunk_bytes = f.read(end - start)

            # decode the bytes into utf-8
            chunk_text = chunk_bytes.decode("utf-8")

            # remove special tokens from chunk_text
            for token in self.special_tokens:
                chunk_text = chunk_text.replace(token, '')

            # find all matches
            matches = re.findall(pattern=PAT, string=chunk_text)

            # get the frequency of all matches
            freq: Dict[Tuple[bytes, ...], int] = {}
            for match in matches:
                token_bytes = tuple(match.encode('utf-8'))
                freq[token_bytes] = freq.get(token_bytes, 0) + 1
            return freq

        with Pool(processes=self.num_workers) as pool:
            results = pool.map(worker, chunks)
        
        # merge results
        total_freq: Dict[Tuple[bytes, ...], int] = {}

        for freq in results:
            for token_tuple, count in freq.items():
                total_freq[token_tuple] = total_freq.get(token_tuple, 0) + count

        return total_freq

    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        self.input_path = input_path
        # intialize vocab
        self._init_vocab()

        # find chunk boundaries
        with open(input_path, 'rb') as f:
            boundaries = self._find_chunk_boundaries(f)

        # pre-tokenize in parallel
        pre_token_freq = self._pre_tokenize_parallel(boundaries)

        # BPE merging process

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
    trainer = BPETokenizerTrainer(vocab_size, special_tokens, num_workers=12)
    return trainer.train(input_path)