import regex as re
from typing import List, Tuple, Dict, BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter
import os
import sys
import gc
import pickle
import hashlib

# Standard BPE tokenization pattern for GPT-2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizerTrainer:
    def __init__(
        self,
        vocab_size: int, 
        special_tokens: List[str],
        num_workers: int = None,
        cache_dir: str = None
    ):
        # parameters
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.cache_dir = cache_dir
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Auto-detect number of workers if not specified
        if num_workers is None:
            available_cpus = cpu_count()
            self.num_workers = min(available_cpus, 32)
            print(f"[Init] Auto-detected {available_cpus} CPUs, using {self.num_workers} workers")
        else:
            self.num_workers = num_workers
            print(f"[Init] Using {self.num_workers} workers (manually specified)")

        # derived parameters
        self.init_vocab_size = 256 + len(special_tokens)
        self.num_merges = vocab_size - self.init_vocab_size
        
        # results
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def _get_cache_key(self, input_path: str) -> str:
        """Generate cache key based on file metadata and special tokens"""
        # Use file size and modification time for speed (instead of hashing entire file)
        stat = os.stat(input_path)
        file_info = f"{stat.st_size}_{stat.st_mtime}"
        
        special_tokens_str = '|'.join(sorted(self.special_tokens))
        cache_key = hashlib.md5(f"{file_info}_{special_tokens_str}".encode()).hexdigest()
        return cache_key
    
    def _save_pretokenization_cache(self, pre_token_freq: Dict, cache_key: str):
        """Save pre-tokenization results to disk"""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, f"pretok_{cache_key}.pkl")
        print(f"[Train] Saving pre-tokenization cache to {cache_file}")
        sys.stdout.flush()
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(pre_token_freq, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Train] Cache saved successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[Train] Warning: Failed to save cache: {e}")
            sys.stdout.flush()
    
    def _load_pretokenization_cache(self, cache_key: str) -> Dict:
        """Load pre-tokenization results from disk if available"""
        if not self.cache_dir:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"pretok_{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        print(f"[Train] Loading pre-tokenization cache from {cache_file}")
        sys.stdout.flush()
        
        try:
            with open(cache_file, 'rb') as f:
                pre_token_freq = pickle.load(f)
            print(f"[Train] Cache loaded: {len(pre_token_freq)} unique tokens, {sum(pre_token_freq.values()):,} total occurrences")
            sys.stdout.flush()
            return pre_token_freq
        except Exception as e:
            print(f"[Train] Warning: Failed to load cache: {e}")
            sys.stdout.flush()
            return None

    def _find_chunk_boundaries(self, file: BinaryIO) -> List[int]:
        """
        Find chunk boundaries aligned to special tokens.
        """
        split_token = self.special_tokens[0].encode("utf-8")
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        chunk_size = max(1, file_size // self.num_workers)
        boundaries = [i * chunk_size for i in range(self.num_workers + 1)]
        boundaries[-1] = file_size
        
        mini_chunk_size = 4096
        
        # For each boundary, slide forward to the nearest special token
        for idx in range(1, len(boundaries) - 1):
            start_guess = boundaries[idx]
            file.seek(start_guess)
            
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    boundaries[idx] = file_size
                    break
                
                found_at = mini_chunk.find(split_token)
                if found_at != -1:
                    boundaries[idx] = start_guess + found_at
                    break
                start_guess += mini_chunk_size
        
        return sorted(set(boundaries))

    def _init_vocab(self):
        """
        Initialize the vocabulary with byte values and special tokens.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode('utf-8')

    def _worker(self, chunk_info: Tuple[str, int, int]) -> Dict[Tuple[bytes, ...], int]:
        """
        Worker function for parallel pre-tokenization.
        Args:
            chunk_info: Tuple of (file_path, start, end) byte offsets.
        Returns:
            A dictionary mapping pre-token tuples to their frequency counts.
        """
        file_path, start, end = chunk_info
        pid = os.getpid()
        chunk_size_mb = (end - start) / (1024 * 1024)
        
        print(f"[Worker PID:{pid}] Starting chunk: bytes {start:,}-{end:,} ({chunk_size_mb:.1f} MB)")
        sys.stdout.flush()

        # get the chunk bytes
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
        
        print(f"[Worker PID:{pid}] Read {len(chunk_bytes):,} bytes, decoding...")
        sys.stdout.flush()

        # decode the bytes into utf-8
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
        print(f"[Worker PID:{pid}] Decoded to {len(chunk_text):,} characters, tokenizing...")
        sys.stdout.flush()

        # remove special tokens from chunk_text
        for token in self.special_tokens:
            chunk_text = chunk_text.replace(token, '')

        # find all matches using findall
        print(f"[Worker PID:{pid}] Finding regex matches...")
        sys.stdout.flush()
        matches = re.findall(pattern=PAT, string=chunk_text)
        total_matches = len(matches)
        print(f"[Worker PID:{pid}] Found {total_matches:,} matches, counting frequencies...")
        sys.stdout.flush()

        # Use Counter for efficient frequency counting
        freq = Counter()
        processed = 0
        report_interval = 1000000

        for match in matches:
            # Convert each character in the match to bytes separately
            # This creates a tuple of bytes objects, not a tuple of ints
            token_bytes = tuple(bytes([b]) for b in match.encode('utf-8'))
            freq[token_bytes] += 1
            processed += 1
            
            if processed % report_interval == 0:
                progress_pct = 100 * processed / total_matches
                print(f"[Worker PID:{pid}]  Progress: {processed:,}/{total_matches:,} tokens ({progress_pct:.1f}%), {len(freq):,} unique tokens so far")
                sys.stdout.flush()
        
        print(f"[Worker PID:{pid}] COMPLETED chunk: {total_matches:,} tokens, {len(freq):,} unique tokens")
        sys.stdout.flush()
        
        return dict(freq)  # Convert Counter to dict for return

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
        print(f"[Main] Created {len(chunks)} chunks for parallel processing with {self.num_workers} workers")
        sys.stdout.flush()

        # Use Counter for efficient merging
        total_freq = Counter()

        with Pool(processes=self.num_workers) as pool:
            print(f"[Main] Starting pool with {self.num_workers} workers...")
            sys.stdout.flush()
            
            # Use imap_unordered for streaming results with chunksize=1 to limit concurrent tasks
            for idx, freq in enumerate(pool.imap_unordered(self._worker, chunks, chunksize=1), 1):
                print(f"[Main] Received result {idx}/{len(chunks)}: {len(freq):,} unique tokens")
                sys.stdout.flush()
                
                # Incremental merge to avoid memory spike
                print(f"[Main]   Merging into totals (before: {len(total_freq):,} unique tokens)...")
                sys.stdout.flush()
                
                total_freq.update(freq)
                
                print(f"[Main]   Merged (after: {len(total_freq):,} unique tokens)")
                sys.stdout.flush()
                
                # Force garbage collection after each merge to free memory
                del freq
                gc.collect()
            
            print(f"[Main] All workers completed!")
            sys.stdout.flush()

        print(f"[Main] Final totals: {len(total_freq):,} unique tokens, {sum(total_freq.values()):,} total occurrences")
        sys.stdout.flush()
        
        # Clean up and return as dict
        result = dict(total_freq)
        del total_freq
        gc.collect()
        
        print(f"[Main] Memory cleanup complete")
        sys.stdout.flush()

        return result

    def train(self, input_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        self.input_path = input_path
        
        # Initialize vocab
        self._init_vocab()
        print(f"[Train] Initialized vocab with {len(self.vocab)} tokens (256 bytes + {len(self.special_tokens)} special tokens)")
        sys.stdout.flush()

        # Generate cache key
        cache_key = self._get_cache_key(input_path) if self.cache_dir else None
        
        # Try to load from cache
        pre_token_freq = None
        if cache_key:
            pre_token_freq = self._load_pretokenization_cache(cache_key)
        
        # If no cache, perform pre-tokenization
        if pre_token_freq is None:
            # Find chunk boundaries
            print(f"[Train] Finding chunk boundaries...")
            sys.stdout.flush()
            with open(input_path, 'rb') as f:
                boundaries = self._find_chunk_boundaries(f)
            print(f"[Train] Found {len(boundaries)} boundaries, creating {len(boundaries)-1} chunks")
            sys.stdout.flush()

            # Pre-tokenize in parallel
            print(f"[Train] Starting parallel pre-tokenization with {self.num_workers} workers...")
            sys.stdout.flush()
            pre_token_freq = self._pre_tokenize_parallel(boundaries)
            print(f"[Train] Pre-tokenization complete: {len(pre_token_freq)} unique tokens, {sum(pre_token_freq.values()):,} total tokens")
            sys.stdout.flush()
            
            # Save to cache
            if cache_key:
                self._save_pretokenization_cache(pre_token_freq, cache_key)

        # Force garbage collection after pre-tokenization
        gc.collect()

        # Initialize pair_to_tuples index and pair_freq
        print(f"[Train] Building pair statistics...")
        sys.stdout.flush()
        pair_to_tuples: Dict[Tuple[bytes, bytes], set] = {}
        pair_freq: Dict[Tuple[bytes, bytes], int] = {}

        processed_tokens = 0
        for token_tuple, freq in pre_token_freq.items():
            processed_tokens += 1
            if processed_tokens % 500000 == 0:
                print(f"[Train]   Processed {processed_tokens:,}/{len(pre_token_freq):,} tokens ({100*processed_tokens/len(pre_token_freq):.1f}%)")
                sys.stdout.flush()
            
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                if pair not in pair_to_tuples:
                    pair_to_tuples[pair] = set()
                pair_to_tuples[pair].add(token_tuple)
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        print(f"[Train] Initial pair statistics: {len(pair_freq):,} unique pairs")
        print(f"[Train] Starting BPE merging: {self.num_merges} merges needed")
        sys.stdout.flush()

        # Force garbage collection after building initial statistics
        gc.collect()

        # BPE merging process
        for merge_iter in range(self.num_merges):
            # Print progress
            if merge_iter % 100 == 0:
                print(f"[Train] Merge iteration {merge_iter}/{self.num_merges}: {len(pre_token_freq):,} tokens, {len(pair_freq):,} pairs")
                sys.stdout.flush()
            
            # Periodic garbage collection and index rebuild to reduce memory fragmentation
            if merge_iter > 0 and merge_iter % 5000 == 0:
                print(f"[Train] Performing memory cleanup at iteration {merge_iter}...")
                sys.stdout.flush()
                
                # Rebuild pair_to_tuples and pair_freq from scratch to reduce fragmentation
                print(f"[Train]   Rebuilding pair index...")
                sys.stdout.flush()
                
                new_pair_to_tuples: Dict[Tuple[bytes, bytes], set] = {}
                new_pair_freq: Dict[Tuple[bytes, bytes], int] = {}
                
                for token_tuple, freq in pre_token_freq.items():
                    for i in range(len(token_tuple) - 1):
                        pair = (token_tuple[i], token_tuple[i + 1])
                        if pair not in new_pair_to_tuples:
                            new_pair_to_tuples[pair] = set()
                        new_pair_to_tuples[pair].add(token_tuple)
                        new_pair_freq[pair] = new_pair_freq.get(pair, 0) + freq
                
                # Replace old structures
                pair_to_tuples = new_pair_to_tuples
                pair_freq = new_pair_freq
                
                # Force garbage collection
                gc.collect()
                print(f"[Train]   Memory cleanup complete: {len(pair_freq):,} pairs")
                sys.stdout.flush()
            
            # Check if no more pairs to merge
            if len(pair_freq) == 0:
                print(f"[Train] No more pairs to merge at iteration {merge_iter}")
                sys.stdout.flush()
                break
            
            # find the most frequent pair in lexicographically greater order
            # key: (frequency, lexicographic order of pair)
            most_frequent_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
            pair_frequency = pair_freq[most_frequent_pair]

            # Print details for first few merges and every 1000th merge
            if merge_iter < 10 or merge_iter % 1000 == 0:
                print(f"[Train]   Merging pair with frequency {pair_frequency:,}, affects {len(pair_to_tuples.get(most_frequent_pair, set()))} token types")
                sys.stdout.flush()

            # merge the most frequent pair
            self.merges.append(most_frequent_pair)
            merged_token = most_frequent_pair[0] + most_frequent_pair[1]
            self.vocab[len(self.vocab)] = merged_token

            # update pre_token_freq, pair_to_tuples and pair_freq with the new merged token
            # only update token_tuples that contain the most_frequent_pair
            affected_tuples = pair_to_tuples.get(most_frequent_pair, set()).copy()

            for token_tuple in affected_tuples:
                freq = pre_token_freq.pop(token_tuple)  # remove old tuple

                # remove old token_tuple from pair_to_tuples and update pair_freq
                for i in range(len(token_tuple) - 1):
                    old_pair = (token_tuple[i], token_tuple[i + 1])
                    if old_pair in pair_to_tuples:
                        pair_to_tuples[old_pair].discard(token_tuple)
                        # decrement the frequency
                        pair_freq[old_pair] -= freq
                        if not pair_to_tuples[old_pair]:
                            del pair_to_tuples[old_pair]
                            del pair_freq[old_pair]

                # create new tuple with merged pair
                new_tuple = []
                i = 0
                while i < len(token_tuple):
                    # if we find the pair, merge it
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == most_frequent_pair:
                        new_tuple.append(merged_token)
                        i += 2  # skip both elements of the pair
                    else:
                        new_tuple.append(token_tuple[i])
                        i += 1

                # convert back to tuple and update frequency
                new_tuple_key = tuple(new_tuple)
                pre_token_freq[new_tuple_key] = pre_token_freq.get(new_tuple_key, 0) + freq

                # add new token_tuple to pair_to_tuples index and update pair_freq
                for i in range(len(new_tuple_key) - 1):
                    new_pair = (new_tuple_key[i], new_tuple_key[i + 1])
                    if new_pair not in pair_to_tuples:
                        pair_to_tuples[new_pair] = set()
                    pair_to_tuples[new_pair].add(new_tuple_key)
                    # increment the frequency
                    pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq

        print(f"[Train] BPE merging complete!")
        print(f"[Train]   Total merges performed: {len(self.merges)}")
        print(f"[Train]   Final vocab size: {len(self.vocab)}")
        print(f"[Train]   Final unique tokens: {len(pre_token_freq):,}")
        sys.stdout.flush()
        
        # Final cleanup
        gc.collect()
        
        return self.vocab, self.merges


def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_workers: int = None,
    cache_dir: str = None
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
        num_workers: Number of worker processes for parallel processing.
            If None, auto-detects based on available CPUs (capped at 32).
        cache_dir: Directory to store pre-tokenization cache.
            If None, no caching is used.

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
        ...     special_tokens=["<|endoftext|>"],
        ...     cache_dir="./bpe_cache"
        ... )
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Number of merges: {len(merges)}")

    Note:
        - The function uses the GPT-2 regex pattern for pre-tokenization
        - Merges do not cross pre-token boundaries or special token boundaries
        - Ties in merge frequency are broken lexicographically (higher pair wins)
        - Pre-tokenization results are cached if cache_dir is provided
    """
    trainer = BPETokenizerTrainer(
        vocab_size, 
        special_tokens, 
        num_workers=num_workers,
        cache_dir=cache_dir
    )
    return trainer.train(input_path)