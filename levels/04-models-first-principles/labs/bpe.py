"""Lab 4a: byte-pair encoding (BPE) from scratch.

Tokenizers turn text into integer IDs. BPE starts from raw bytes and
repeatedly merges the most frequent adjacent pair into a new token.
GPT-2, Llama 3 and most modern LLMs use byte-level BPE variants.

    python3 bpe.py
"""

import re
from collections import Counter

# Like GPT-2, split text into chunks first (words with their leading space,
# numbers, punctuation) and only merge inside a chunk, so tokens never span words.
PRETOKENIZE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")


def pair_counts(ids: list[int]) -> Counter:
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        words = Counter(PRETOKENIZE.findall(text))
        chunks = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            counts = Counter()
            for w, ids in chunks.items():
                for pair, c in pair_counts(ids).items():
                    counts[pair] += c * words[w]
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break
            chunks = {w: merge(ids, pair, new_id) for w, ids in chunks.items()}
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # apply the earliest-learned merge first, as in training
            pair = min(pair_counts(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in PRETOKENIZE.findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


CORPUS = """
The GPU reads the model weights from memory for every token it generates.
The KV cache stores the keys and values of every previous token, so the model
does not recompute them. Inference engines batch many requests together so the
weights are read once per step for the whole batch. Continuous batching adds and
removes requests at every step. PagedAttention stores the cache in fixed-size
blocks, like pages of virtual memory. Prefix caching reuses the blocks of a shared
system prompt. Quantization stores the weights in fewer bits, which reduces the
memory the GPU must read for every token. Speculative decoding lets a small model
guess several tokens that the large model then verifies in a single step.
The scheduler decides which requests run in each step and which requests wait.
"""


def main():
    tok = BPE()
    tok.train(CORPUS, vocab_size=400)
    for text in ["The KV cache stores tokens.", "नमस्ते दुनिया", "def f(x): return x**2", "Inference!!!"]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        print(f"{len(text.encode()):3d} bytes -> {len(ids):3d} tokens  {text!r}")
        if text.isascii():
            print("      pieces:", [tok.vocab[i].decode() for i in ids])
    print("\nText that looks like the training corpus compresses well; other scripts and code do not."
          "\nThat is why the same sentence costs different numbers of tokens in different languages,"
          "\nand why token count, not character count, drives latency and cost.")


if __name__ == "__main__":
    main()
