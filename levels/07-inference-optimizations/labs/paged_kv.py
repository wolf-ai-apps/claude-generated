"""Lab 7b: PagedAttention's memory manager and prefix caching.

    python3 paged_kv.py

vLLM stores the KV cache in fixed-size blocks (default 16 tokens). Each sequence
has a block table mapping its logical blocks to physical blocks, exactly like
an OS page table maps virtual pages to physical frames. This gives:
  - no fragmentation from reserving max_len per request
  - sharing: parallel samples and common prefixes point at the same blocks
  - copy-on-write when a shared block is about to be modified
Blocks here store token IDs instead of real K/V tensors; the bookkeeping is the same.
"""

import hashlib
from collections import OrderedDict


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free = list(range(num_blocks))[::-1]
        self.ref = [0] * num_blocks
        self.data = [[] for _ in range(num_blocks)]        # stands in for K/V tensors
        self.cached: OrderedDict[str, int] = OrderedDict()  # prefix hash -> block (LRU order)
        self.hash_of: dict[int, str] = {}

    def num_free(self):
        return len(self.free) + sum(1 for b in self.cached.values() if self.ref[b] == 0)

    def allocate(self) -> int:
        if not self.free:
            self._evict()
        b = self.free.pop()
        self.ref[b] = 1
        self.data[b] = []
        return b

    def _evict(self):
        for h, b in self.cached.items():                  # oldest first
            if self.ref[b] == 0:
                del self.cached[h]
                del self.hash_of[b]
                self.free.append(b)
                return
        raise OutOfBlocks("no free or evictable blocks: the scheduler must preempt a sequence")

    def incref(self, b):
        self.ref[b] += 1

    def decref(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0 and b not in self.hash_of:
            self.free.append(b)                            # cached blocks stay until evicted

    def register(self, h: str, b: int):
        self.cached[h] = b
        self.hash_of[b] = h

    def lookup(self, h: str):
        b = self.cached.get(h)
        if b is not None:
            self.cached.move_to_end(h)
        return b


def block_hash(prev: str, tokens) -> str:
    """A block's identity is its tokens AND everything before it (hash chain)."""
    return hashlib.sha256((prev + "|" + ",".join(map(str, tokens))).encode()).hexdigest()[:16]


class Sequence:
    def __init__(self, alloc: BlockAllocator):
        self.a = alloc
        self.table: list[int] = []   # logical block index -> physical block
        self.length = 0
        self.cached_tokens = 0

    def _last_hash(self, upto_block):
        h = ""
        for i in range(upto_block):
            h = block_hash(h, self.a.data[self.table[i]])
        return h

    def prefill(self, tokens, use_prefix_cache=True):
        """Allocate blocks for the prompt, reusing cached full blocks for a matching prefix."""
        bs = self.a.block_size
        h, i = "", 0
        if use_prefix_cache:
            while i + bs <= len(tokens):
                nh = block_hash(h, tokens[i:i + bs])
                b = self.a.lookup(nh)
                if b is None:
                    break
                self.a.incref(b)
                self.table.append(b)
                h, i = nh, i + bs
            self.cached_tokens = i                         # these tokens need no compute
            self.length = i
        for t in tokens[i:]:
            self.append(t, register=use_prefix_cache)

    def append(self, token, register=True):
        bs = self.a.block_size
        if self.length % bs == 0:                          # need a new block
            self.table.append(self.a.allocate())
        b = self.table[-1]
        if self.a.ref[b] > 1:                              # shared: copy on write
            nb = self.a.allocate()
            self.a.data[nb] = list(self.a.data[b])
            self.a.decref(b)
            self.table[-1] = b = nb
        self.a.data[b].append(token)
        self.length += 1
        if register and self.length % bs == 0:             # block became full: make it reusable
            idx = len(self.table) - 1
            self.a.register(block_hash(self._last_hash(idx), self.a.data[b]), b)

    def fork(self) -> "Sequence":
        """Share all blocks with a child (e.g. n>1 parallel samples, beam search)."""
        child = Sequence(self.a)
        child.table = list(self.table)
        child.length = self.length
        for b in self.table:
            self.a.incref(b)
        return child

    def tokens(self):
        return [t for b in self.table for t in self.a.data[b]]

    def free(self):
        for b in self.table:
            self.a.decref(b)
        self.table = []


def contiguous_waste(lengths, max_len):
    """Reserving max_len tokens per request up front (pre-PagedAttention servers)."""
    used = sum(lengths)
    return 1 - used / (len(lengths) * max_len)


def paged_waste(lengths, block_size):
    used = sum(lengths)
    allocated = sum(-(-n // block_size) * block_size for n in lengths)
    return 1 - used / allocated


def main():
    import random
    rng = random.Random(0)
    lengths = [rng.randint(50, 1500) for _ in range(100)]
    print("1) Fragmentation for 100 requests with 50-1500 tokens each:")
    print(f"   contiguous, reserve max_len=2048: {contiguous_waste(lengths, 2048):.0%} of reserved KV memory wasted")
    print(f"   paged, 16-token blocks:           {paged_waste(lengths, 16):.1%} wasted (only the last block is partial)")

    a = BlockAllocator(num_blocks=64, block_size=4)
    system = list(range(100, 112))                         # 12-token shared system prompt
    s1 = Sequence(a)
    s1.prefill(system + [1, 2, 3])
    s2 = Sequence(a)
    s2.prefill(system + [7, 8])
    print("\n2) Prefix caching with 4-token blocks and a 12-token shared system prompt:")
    print(f"   request 1 block table {s1.table}, cached tokens {s1.cached_tokens}")
    print(f"   request 2 block table {s2.table}, cached tokens {s2.cached_tokens}  <- first 3 blocks shared")

    parent = Sequence(a)
    parent.prefill([5, 5, 5, 5, 6, 6], use_prefix_cache=False)
    child = parent.fork()
    print("\n3) Fork for parallel sampling, then copy-on-write:")
    print(f"   after fork:   parent {parent.table}  child {child.table}")
    parent.append(1)
    child.append(2)
    print(f"   after append: parent {parent.table}  child {child.table}  (last block copied)")
    print(f"   parent tokens {parent.tokens()}  child tokens {child.tokens()}")


if __name__ == "__main__":
    main()
