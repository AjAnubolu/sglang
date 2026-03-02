from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List, Optional

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.benchmark.datasets.random import sample_random_requests


@dataclass
class EmbeddingDataset(BaseDataset):
    input_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str

    @classmethod
    def from_args(cls, args: Namespace) -> "EmbeddingDataset":
        return cls(
            input_len=args.random_input_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
        )

    def load(self, tokenizer: Any, model_id: Optional[str] = None) -> List[DatasetRow]:
        rows = sample_random_requests(
            input_len=self.input_len,
            output_len=1,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
        )
        for row in rows:
            row.output_len = 0
        return rows
