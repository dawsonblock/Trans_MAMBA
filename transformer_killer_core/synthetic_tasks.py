"""synthetic_tasks.py

Synthetic benchmark tasks for testing memory capabilities.

Tasks:
    - CopyMemoryDataset: Copy tokens after a delay
    - AssocRecallDataset: Key-value associative recall
    - SelectiveCopyDataset: Copy only marked tokens (NEW)
    - InductionHeadDataset: Pattern completion (NEW)
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset


class CopyMemoryDataset(Dataset):
    """Classic copy-memory task - tests long-range memory.

    Sequence layout (length = seq_len):

      [x_1, ..., x_L, 0, ..., 0, 9, 0, ..., 0]

    - x_i are random tokens in [1, vocab_size-2]
    - 0 is the blank/pad token
    - 9 is a delimiter / "go" token (must be < vocab_size)

    Targets:

      - 0 everywhere except the last L positions, where the model must
        output x_1..x_L (copied from the prefix) after a fixed delay.

    We treat token 0 as "ignore" in the loss.
    """

    def __init__(self, seq_len: int, delay: int,
                 num_samples: int = 10000,
                 vocab_size: int = 10,
                 copy_length: int = 10,
                 device: str = "cpu"):
        assert vocab_size >= 10, "vocab_size must be >= 10 for delimiter"
        super().__init__()
        self.seq_len = seq_len
        self.delay = delay
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.copy_length = copy_length
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self.copy_length
        D = self.delay
        T = self.seq_len

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)

        # Sample sequence to remember (avoid using 0 and delimiter)
        data = torch.randint(low=1, high=self.vocab_size - 1, size=(L,))

        # Place data at the beginning
        x[:L] = data

        # Delimiter position
        delim_pos = min(L + D, T - 1)
        x[delim_pos] = self.vocab_size - 1  # delimiter token (e.g., 9)

        # Targets: copy data after delimiter
        start_out = delim_pos + 1
        end_out = min(start_out + L, T)
        length = end_out - start_out
        if length > 0:
            y[start_out:end_out] = data[:length]

        return x, y

    def get_difficulty(self) -> dict:
        """Return task difficulty metrics."""
        return {
            "task": "copy_memory",
            "delay": self.delay,
            "copy_length": self.copy_length,
            "total_length": self.seq_len,
            "memory_span": self.delay + self.copy_length,
        }


class AssocRecallDataset(Dataset):
    """Associative recall task - tests content-addressable memory.

    Sequence layout (length = seq_len):

      [k1, v1, k2, v2, ..., kN, vN, 9, q, 0, 0, ...]

    - keys/values are random tokens in [1, vocab_size-2]
    - 9 is a delimiter token
    - q is one of the keys; the model must output the associated value
      at the position after q.
    - 0 is pad (ignored in the loss).
    """

    def __init__(self, seq_len: int,
                 num_pairs: int,
                 num_samples: int = 10000,
                 vocab_size: int = 16,
                 device: str = "cpu"):
        assert vocab_size >= 10, "vocab_size must be >= 10 for delimiter"
        super().__init__()
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        N = self.num_pairs

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)

        # Sample keys/values (avoid 0 and delimiter)
        keys = torch.randint(low=1, high=self.vocab_size - 1, size=(N,))
        vals = torch.randint(low=1, high=self.vocab_size - 1, size=(N,))

        # Layout key/value pairs
        pos = 0
        for i in range(N):
            if pos + 2 >= T:
                break
            x[pos] = keys[i]
            x[pos + 1] = vals[i]
            pos += 2

        # Delimiter
        if pos < T:
            x[pos] = self.vocab_size - 1  # delimiter
            pos += 1

        # Query
        if pos < T:
            q_idx = torch.randint(low=0, high=N, size=(1,)).item()
            q_key = keys[q_idx]
            q_val = vals[q_idx]
            x[pos] = q_key
            # Target at next position
            if pos + 1 < T:
                y[pos + 1] = q_val

        return x, y

    def get_difficulty(self) -> dict:
        """Return task difficulty metrics."""
        return {
            "task": "assoc_recall",
            "num_pairs": self.num_pairs,
            "total_length": self.seq_len,
            "distractor_ratio": (self.num_pairs - 1) / self.num_pairs,
        }


class SelectiveCopyDataset(Dataset):
    """Selective copy task - copy only marked tokens.

    Sequence layout:
        [x1, m1, x2, m2, ..., xN, mN, DELIM, 0, 0, ...]

    where mi ∈ {0, 1} is a marker. Model must output xi where mi=1.
    Tests selective attention and filtering.
    """

    def __init__(self, seq_len: int,
                 num_tokens: int = 10,
                 select_ratio: float = 0.3,
                 num_samples: int = 10000,
                 vocab_size: int = 16,
                 device: str = "cpu"):
        super().__init__()
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.select_ratio = select_ratio
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        N = self.num_tokens

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)

        # Generate tokens and markers
        tokens = torch.randint(1, self.vocab_size - 2, size=(N,))
        markers = (torch.rand(N) < self.select_ratio).long()

        # Layout: token, marker pairs
        pos = 0
        selected = []
        for i in range(N):
            if pos + 2 >= T - 2:
                break
            x[pos] = tokens[i]
            x[pos + 1] = markers[i]  # 0 or 1
            if markers[i] == 1:
                selected.append(tokens[i])
            pos += 2

        # Delimiter
        x[pos] = self.vocab_size - 1
        pos += 1

        # Targets: selected tokens
        for i, tok in enumerate(selected):
            if pos + i < T:
                y[pos + i] = tok

        return x, y

    def get_difficulty(self) -> dict:
        return {
            "task": "selective_copy",
            "num_tokens": self.num_tokens,
            "select_ratio": self.select_ratio,
            "expected_selected": int(self.num_tokens * self.select_ratio),
        }


class InductionHeadDataset(Dataset):
    """Induction head task - pattern completion.

    Sequence layout:
        [A, B, ..., A, ?]

    Model must predict B when it sees A again.
    Tests in-context learning and pattern matching.
    """

    def __init__(self, seq_len: int,
                 pattern_len: int = 2,
                 num_samples: int = 10000,
                 vocab_size: int = 16,
                 device: str = "cpu"):
        super().__init__()
        self.seq_len = seq_len
        self.pattern_len = pattern_len
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        P = self.pattern_len

        x = torch.zeros(T, dtype=torch.long)
        y = torch.zeros(T, dtype=torch.long)

        # Generate pattern [A, B] to complete
        pattern = torch.randint(1, self.vocab_size - 1, size=(P,))

        # Fill with random tokens
        x[:] = torch.randint(1, self.vocab_size - 1, size=(T,))

        # Insert pattern at random early position
        insert_pos = torch.randint(0, T // 3, size=(1,)).item()
        x[insert_pos:insert_pos + P] = pattern

        # Insert pattern trigger near end (A only)
        trigger_pos = T - P - 1
        x[trigger_pos:trigger_pos + P - 1] = pattern[:-1]

        # Target: complete the pattern (B)
        y[trigger_pos + P - 1] = pattern[-1]

        return x, y

    def get_difficulty(self) -> dict:
        return {
            "task": "induction_head",
            "pattern_len": self.pattern_len,
            "total_length": self.seq_len,
            "context_distance": self.seq_len - self.seq_len // 3,
        }


def get_task_dataset(task: str, **kwargs) -> Dataset:
    """Factory function for task datasets.

    Args:
        task: One of 'copy_memory', 'assoc_recall', 'selective_copy', 'induction'
        **kwargs: Task-specific parameters

    Returns:
        Dataset instance
    """
    tasks = {
        "copy_memory": CopyMemoryDataset,
        "assoc_recall": AssocRecallDataset,
        "selective_copy": SelectiveCopyDataset,
        "induction": InductionHeadDataset,
    }
    if task not in tasks:
        raise ValueError(f"Unknown task: {task}. Choose from {list(tasks.keys())}")
    return tasks[task](**kwargs)
