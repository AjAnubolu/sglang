"""
Benchmark the throughput of embedding models in the offline mode.
It accepts server arguments (the same as launch_server.py) and benchmark arguments.

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_embedding_offline_throughput --model-path BAAI/bge-large-en-v1.5 --num-prompts 1000

## Random dataset with default args
python -m sglang.bench_embedding_offline_throughput --model-path BAAI/bge-large-en-v1.5 --dataset-name random --random-input-len 512
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import time
from typing import List, Optional

import numpy as np

from sglang.benchmark.datasets import DatasetRow, get_dataset, sample_random_requests
from sglang.benchmark.utils import get_tokenizer, set_ulimit
from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 512
    random_range_ratio: float = 0.0
    seed: int = 1
    skip_warmup: bool = False
    profile: bool = False
    apply_chat_template: bool = False
    prompt_suffix: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. "
            "Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input length, "
            "used only for random dataset.",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all prompts.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[DatasetRow],
    profile: bool,
):
    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r.prompt_len for r in reqs),
        "request_throughput": -1,
        "input_throughput": -1,
    }

    prompt = [r.prompt for r in reqs]

    if profile:
        assert (
            "SGLANG_TORCH_PROFILER_DIR" in os.environ
        ), "Please set SGLANG_TORCH_PROFILER_DIR."
        os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
        backend.start_profile()

    st = time.perf_counter()
    encode_out = backend.encode(prompt=prompt)
    latency = time.perf_counter() - st

    if profile:
        dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
        known_files = set(os.listdir(dir))
        backend.stop_profile()
        monitor_trace_file(known_files, dir)

    if backend_name == "runtime":
        encode_out = json.loads(encode_out)

    measurement_results["total_latency"] = latency
    measurement_results["request_throughput"] = (
        measurement_results["successful_requests"] / latency
    )
    measurement_results["input_throughput"] = (
        measurement_results["total_input_tokens"] / latency
    )

    return measurement_results


def monitor_trace_file(known_files, directory, interval=1):
    print(f"Monitoring {directory} for new trace files...")

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    else:
        raise ValueError('Please set backend to either "engine" or "runtime"')

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environments
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Read dataset
    # get_dataset requires sharegpt_output_len and random_output_len attributes.
    # For embeddings we don't need output tokens, so set dummy values.
    bench_args_with_output = argparse.Namespace(**dataclasses.asdict(bench_args))
    bench_args_with_output.sharegpt_output_len = None
    bench_args_with_output.random_output_len = 1
    input_requests = get_dataset(bench_args_with_output, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=1,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    # Warm up
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            profile=False,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        profile=bench_args.profile,
    )
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print(
        "\n{s:{c}^{n}}".format(
            s=" Offline Embedding Throughput Benchmark Result ", n=60, c="="
        )
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Force embedding mode
    args.is_embedding = True

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)
