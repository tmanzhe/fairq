"""Compare FIFO and fair dispatch with a simulated worker pool, no sleeping.

Adapted from infergate's MIT-licensed scripts/fair_loadtest.py by Truman Chan.
All arrivals are at time zero. Metrics are computed from dispatched completions,
not wall-clock service measurements. No server, keys, or model APIs are needed.
"""

import argparse
import heapq
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from . import FIFOQueue, FairQueue, Request
from ._queues import _positive_integer


def workload() -> list[Request[str]]:
    """One noisy burst first, followed by three small tenants."""
    return [
        Request(tenant, f"{tenant}-{index}", (50, 100, 150, 200)[index % 4])
        for tenant, count in (
            ("noisy", 1000), ("small-1", 20), ("small-2", 20), ("small-3", 20)
        )
        for index in range(count)
    ]


def simulate[T](
    requests: Iterable[Request[T]],
    queue: FIFOQueue[T] | FairQueue[T],
    workers: int = 4,
    tokens_per_second: int = 1000,
) -> dict[str, Any]:
    """Measure simulated completion latency, including queue wait and service.

    A request occupies a free worker for cost / token rate seconds. Integer
    token ticks avoid clock noise. The supplied queue must start empty.
    """
    _positive_integer(workers, "workers")
    _positive_integer(tokens_per_second, "tokens_per_second")
    if queue:
        raise ValueError("simulation requires an empty queue")
    for request in requests:
        queue.enqueue(request.tenant, request.item, cost=request.cost)
    available = [(0, worker) for worker in range(workers)]
    heapq.heapify(available)
    latencies: dict[str, list[float]] = defaultdict(list)
    elapsed_ticks = 0
    while queue:
        started, worker = heapq.heappop(available)
        request = queue.dispatch()
        finished = started + request.cost
        heapq.heappush(available, (finished, worker))
        elapsed_ticks = max(elapsed_ticks, finished)
        latencies[request.tenant].append(finished / tokens_per_second * 1000)
    count = sum(len(values) for values in latencies.values())
    elapsed = elapsed_ticks / tokens_per_second

    def percentile(values: list[float], fraction: float) -> float:
        return round(sorted(values)[math.ceil(len(values) * fraction) - 1], 3)

    return {
        "requests": count,
        "simulated_seconds": elapsed,
        "throughput_requests_per_second": round(count / elapsed, 3) if elapsed else 0,
        "tenants": {
            tenant: {
                "requests": len(values),
                "p50_latency_ms": percentile(values, 0.5),
                "p95_latency_ms": percentile(values, 0.95),
            }
            for tenant, values in sorted(latencies.items())
        },
    }


def run_comparison(
    workers: int = 4, weights: Mapping[str, int] | None = None
) -> dict[str, Any]:
    requests = workload()
    return {
        "mode": "simulated workload (discrete-event, no network, $0)",
        "workers": workers,
        "tokens_per_second_per_worker": 1000,
        "arrival": "all at time zero, noisy tenant first",
        "estimated_tokens": [50, 100, 150, 200],
        "quantum_tokens": 100,
        "weights": {
            tenant: (weights if weights is not None else {}).get(tenant, 1)
            for tenant in ("noisy", "small-1", "small-2", "small-3")
        },
        "fifo": simulate(requests, FIFOQueue(), workers),
        "fair": simulate(requests, FairQueue(quantum=100, weights=weights), workers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = run_comparison(args.workers)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("Simulated workload: 1 noisy tenant x 1,000 requests; 3 small tenants x 20.")
    print("All arrive at t=0, noisy first; costs cycle 50/100/150/200 estimated tokens.")
    print(f"{args.workers} workers x 1,000 tokens/s; quantum=100; all weights=1.")
    print("Completion latency includes queue wait + service; nearest-rank percentiles.")
    print()
    print("tenant     requests   FIFO p50/p95 (ms)    Fair p50/p95 (ms)")
    for tenant, fifo in report["fifo"]["tenants"].items():
        fair = report["fair"]["tenants"][tenant]
        print(
            f"{tenant:<10} {fifo['requests']:>8}   "
            f"{fifo['p50_latency_ms']:>7,.0f} / {fifo['p95_latency_ms']:>7,.0f}   "
            f"{fair['p50_latency_ms']:>7,.0f} / {fair['p95_latency_ms']:>7,.0f}"
        )
    print()
    for mode in ("fifo", "fair"):
        result = report[mode]
        print(
            f"{mode.upper()}: {result['requests']:,} completed in "
            f"{result['simulated_seconds']:g} simulated s; "
            f"throughput={result['throughput_requests_per_second']:.3f} requests/s"
        )
    print("Simulated results only: no scheduling overhead, batching, or estimation error.")


if __name__ == "__main__":
    main()
