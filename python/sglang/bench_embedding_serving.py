"""
Benchmark online serving throughput for embedding models.

Usage:
python3 -m sglang.bench_embedding_serving --num-prompts 1000

python3 -m sglang.bench_embedding_serving --dataset-name random --num-prompts 3000 --random-input-len 512
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from sglang.benchmark.datasets import DatasetRow, get_dataset
from sglang.benchmark.utils import get_tokenizer, set_ulimit

global args


def _create_bench_client_session():
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB
    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


def get_auth_headers() -> Dict[str, str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}
    else:
        api_key = os.environ.get("API_KEY")
        if api_key:
            return {"Authorization": f"{api_key}"}
        return {}


@dataclass
class EmbeddingRequestOutput:
    success: bool = False
    latency: float = 0.0
    prompt_len: int = 0
    error: str = ""


async def async_request_embedding(
    api_url: str,
    model: str,
    prompt: str,
    prompt_len: int,
    pbar: Optional[tqdm] = None,
) -> EmbeddingRequestOutput:
    headers = get_auth_headers()
    payload = {
        "model": model,
        "input": prompt,
    }

    output = EmbeddingRequestOutput(prompt_len=prompt_len)

    async with _create_bench_client_session() as session:
        st = time.perf_counter()
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    await response.json()
                    output.latency = time.perf_counter() - st
                    output.success = True
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    input_requests_iter = iter(input_requests)
    for request in input_requests_iter:
        yield request

        if request_rate == float("inf"):
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


async def benchmark(
    api_url: str,
    model: str,
    input_requests: List[DatasetRow],
    request_rate: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
):
    # Limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request(prompt, prompt_len, pbar):
        if semaphore is None:
            return await async_request_embedding(
                api_url, model, prompt, prompt_len, pbar
            )
        async with semaphore:
            return await async_request_embedding(
                api_url, model, prompt, prompt_len, pbar
            )

    # Warmup
    print("Starting warmup...")
    test_request = input_requests[0]
    warmup_output = await async_request_embedding(
        api_url, model, test_request.prompt, test_request.prompt_len
    )
    if not warmup_output.success:
        raise ValueError(
            "Warmup failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {warmup_output.error}"
        )
    print("Warmup completed. Starting main benchmark run...")
    time.sleep(1.0)

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    async for request in get_request(input_requests, request_rate):
        tasks.append(
            asyncio.create_task(
                limited_request(request.prompt, request.prompt_len, pbar)
            )
        )

    outputs: List[EmbeddingRequestOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Compute metrics
    completed = 0
    total_input_tokens = 0
    e2e_latencies: List[float] = []

    for output in outputs:
        if output.success:
            completed += 1
            total_input_tokens += output.prompt_len
            e2e_latencies.append(output.latency)
        else:
            if args.verbose:
                print(f"Request failed: {output.error}")

    if completed == 0:
        print(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments."
        )
        return {}

    request_throughput = completed / benchmark_duration
    input_throughput = total_input_tokens / benchmark_duration
    mean_e2e_latency_ms = np.mean(e2e_latencies) * 1000
    median_e2e_latency_ms = np.median(e2e_latencies) * 1000
    p99_e2e_latency_ms = np.percentile(e2e_latencies, 99) * 1000

    result = {
        "backend": args.backend,
        "successful_requests": completed,
        "total_requests": len(input_requests),
        "benchmark_duration": benchmark_duration,
        "total_input_tokens": total_input_tokens,
        "request_throughput": request_throughput,
        "input_throughput": input_throughput,
        "request_rate": request_rate if request_rate != float("inf") else "inf",
        "max_concurrency": max_concurrency,
        "mean_e2e_latency_ms": mean_e2e_latency_ms,
        "median_e2e_latency_ms": median_e2e_latency_ms,
        "p99_e2e_latency_ms": p99_e2e_latency_ms,
    }

    # Print results
    print("\n{s:{c}^{n}}".format(s=" Embedding Serving Benchmark Result ", n=60, c="="))
    print("{:<40} {:<10}".format("Backend:", args.backend))
    print(
        "{:<40} {:<10}".format(
            "Successful requests:", f"{completed}/{len(input_requests)}"
        )
    )
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", total_input_tokens))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", request_throughput))
    print(
        "{:<40} {:<10.2f}".format("Input token throughput (tok/s):", input_throughput)
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=60, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", median_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", p99_e2e_latency_ms))
    print("=" * 60)

    # Save results
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        output_file_name = f"embedding_{args.backend}_{now}_{args.num_prompts}.jsonl"
    with open(output_file_name, "a") as fout:
        fout.write(json.dumps(result) + "\n")
    print(f"\nResults saved to {output_file_name}")

    return result


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    print(f"benchmark_args={args}")

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url
    if args.port is None:
        args.port = 30000

    api_url = (
        f"{args.base_url}/v1/embeddings"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/embeddings"
    )
    base_url = (
        f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
    )

    # Get model name
    if args.model is None:
        import requests

        model_url = f"{base_url}/v1/models"
        try:
            response = requests.get(model_url, headers=get_auth_headers())
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id)

    # get_dataset requires sharegpt_output_len and random_output_len attributes.
    # For embeddings we don't need output tokens, so set dummy values.
    args.sharegpt_output_len = None
    args.random_output_len = 1

    input_requests = get_dataset(args, tokenizer)

    return asyncio.run(
        benchmark(
            api_url=api_url,
            model=args.model,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark online serving throughput for embedding models."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Default is 30000.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name or path of the model. If not set, will query /v1/models.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of the tokenizer. If not set, uses the model.",
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
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=512,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input length, used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file name.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print errors for failed requests.",
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

    args = parser.parse_args()
    run_benchmark(args)
