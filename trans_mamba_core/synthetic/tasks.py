"""
Synthetic benchmark tasks for memory evaluation.

Tasks:
- CopyMemory: Memorize tokens, recall after delay
- AssocRecall: Key-value association lookup
- SelectiveCopy: Copy only marked tokens
- InductionHead: Pattern completion (A B ... A -> B)
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticTaskConfig:
    """Configuration for synthetic tasks."""

    seq_len: int = 128
    vocab_size: int = 16
    num_samples: int = 50000
    delay: int = 40
    copy_length: int = 10
    num_pairs: int = 4
    num_markers: int = 5
    pattern_len: int = 2


class CopyMemoryDataset(Dataset):
    """
    Copy memory task: memorize tokens, then recall after delay.

    Sequence: [tokens] [delim] [blanks...] [delim] [recall]
    Target: zeros except at recall positions where original tokens appear.
    """

    def __init__(
        self,
        seq_len: int = 128,
        delay: int = 40,
        vocab_size: int = 16,
        num_samples: int = 50000,
        copy_length: int = 10,
    ):
        self.seq_len = seq_len
        self.delay = delay
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.copy_length = copy_length

        self.blank_token = 0
        self.delim_token = vocab_size - 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        L = self.copy_length
        D = self.delay
        T = self.seq_len

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)
        mask = torch.zeros(T, dtype=torch.float)

        data = torch.randint(1, self.vocab_size - 1, (L,))
        x[:L] = data

        delim_pos = min(L + D, T - 1)
        x[delim_pos] = self.delim_token

        start_out = delim_pos + 1
        end_out = min(start_out + L, T)
        length = end_out - start_out

        if length > 0:
            y[start_out:end_out] = data[:length]
            mask[start_out:end_out] = 1.0

        return x, y, mask


class AssocRecallDataset(Dataset):
    """
    Associative recall: key-value pairs, then query a key.

    Sequence: [k1 v1] [k2 v2] ... [kN vN] [delim] [query_k] [?]
    Target: zeros except at answer position where value appears.
    """

    def __init__(
        self,
        seq_len: int = 64,
        num_pairs: int = 4,
        vocab_size: int = 16,
        num_samples: int = 50000,
    ):
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.delim_token = vocab_size - 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        T = self.seq_len
        N = self.num_pairs

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)
        mask = torch.zeros(T, dtype=torch.float)

        keys = torch.randint(1, self.vocab_size - 1, (N,))
        vals = torch.randint(1, self.vocab_size - 1, (N,))

        pos = 0
        for i in range(N):
            if pos + 2 >= T:
                break
            x[pos] = keys[i]
            x[pos + 1] = vals[i]
            pos += 2

        if pos < T:
            x[pos] = self.delim_token
            pos += 1

        if pos < T:
            q_idx = torch.randint(0, N, (1,)).item()
            x[pos] = keys[q_idx]
            if pos + 1 < T:
                y[pos + 1] = vals[q_idx]
                mask[pos + 1] = 1.0

        return x, y, mask


class SelectiveCopyDataset(Dataset):
    """
    Selective copy: copy only tokens at marked positions.

    Sequence: [t1 m1 t2 m2 ... tN mN] [delim] [recall]
    Markers indicate which tokens to copy.
    """

    def __init__(
        self,
        seq_len: int = 128,
        num_markers: int = 5,
        vocab_size: int = 16,
        num_samples: int = 50000,
    ):
        self.seq_len = seq_len
        self.num_markers = num_markers
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        self.blank_token = 0
        self.delim_token = vocab_size - 1
        self.marker_token = vocab_size - 2

        self.content_len = (seq_len - 1 - num_markers) // 2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        T = self.seq_len
        L = self.content_len
        M = self.num_markers

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)
        mask = torch.zeros(T, dtype=torch.float)

        content = torch.randint(1, self.vocab_size - 2, (L,))
        marker_pos = torch.randperm(L)[:M].sort()[0]
        markers = torch.zeros(L, dtype=torch.long)
        markers[marker_pos] = 1

        for i in range(L):
            x[i * 2] = content[i]
            if markers[i]:
                x[i * 2 + 1] = self.marker_token
            else:
                x[i * 2 + 1] = self.blank_token

        delim_pos = L * 2
        x[delim_pos] = self.delim_token

        recall_start = delim_pos + 1
        marked_tokens = content[marker_pos]

        for i, tok in enumerate(marked_tokens):
            pos = recall_start + i
            if pos < T:
                y[pos] = tok
                mask[pos] = 1.0

        return x, y, mask


class InductionHeadDataset(Dataset):
    """
    Induction head: pattern completion (A B ... A -> B).

    Tests ability to match patterns seen earlier in sequence.
    """

    def __init__(
        self,
        seq_len: int = 128,
        pattern_len: int = 2,
        vocab_size: int = 16,
        num_samples: int = 50000,
    ):
        self.seq_len = seq_len
        self.pattern_len = pattern_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        T = self.seq_len
        P = self.pattern_len

        x = torch.randint(1, self.vocab_size - 1, (T,))
        y = torch.zeros(T, dtype=torch.long)
        mask = torch.zeros(T, dtype=torch.float)

        pattern = torch.randint(1, self.vocab_size - 1, (P,))

        first_pos = torch.randint(0, T // 3, (1,)).item()
        x[first_pos:first_pos + P] = pattern

        second_pos = torch.randint(T // 2, T - P - 1, (1,)).item()
        x[second_pos:second_pos + P - 1] = pattern[:-1]

        y[second_pos + P - 1] = pattern[-1]
        mask[second_pos + P - 1] = 1.0

        return x, y, mask


def get_dataset(task_name: str, cfg: SyntheticTaskConfig) -> Dataset:
    """Factory function for synthetic datasets."""

    datasets = {
        "copy_memory": lambda: CopyMemoryDataset(
            seq_len=cfg.seq_len,
            delay=cfg.delay,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
            copy_length=cfg.copy_length,
        ),
        "assoc_recall": lambda: AssocRecallDataset(
            seq_len=cfg.seq_len,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
        ),
        "selective_copy": lambda: SelectiveCopyDataset(
            seq_len=cfg.seq_len,
            num_markers=cfg.num_markers,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
        ),
        "induction_head": lambda: InductionHeadDataset(
            seq_len=cfg.seq_len,
            pattern_len=cfg.pattern_len,
            vocab_size=cfg.vocab_size,
            num_samples=cfg.num_samples,
        ),
    }

    if task_name not in datasets:
        raise ValueError(f"Unknown task: {task_name}")

    return datasets[task_name]()
