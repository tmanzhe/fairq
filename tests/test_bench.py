import json
import subprocess
import sys

import pytest

from fairq import FIFOQueue, FairQueue, Request
from fairq.bench import run_comparison, simulate


def test_mock_workers_measure_completion_including_wait_and_stay_busy():
    requests = [Request("a", str(index), 100) for index in range(5)]
    report = simulate(requests, FIFOQueue(), workers=2, tokens_per_second=1000)
    assert report == {
        "requests": 5,
        "simulated_seconds": 0.3,
        "throughput_requests_per_second": 16.667,
        "tenants": {"a": {"requests": 5, "p50_latency_ms": 200, "p95_latency_ms": 300}},
    }


def test_latency_percentiles_use_completion_time_not_dispatch_order():
    # The first request finishes last. With 20 requests, p95 is the 19th finish.
    requests = [Request("a", "slow", 1000)]
    requests += [Request("a", str(index), 1) for index in range(19)]
    report = simulate(requests, FIFOQueue(), workers=20)
    assert report["tenants"]["a"]["p50_latency_ms"] == 1
    assert report["tenants"]["a"]["p95_latency_ms"] == 1
    assert report["simulated_seconds"] == 1


@pytest.mark.parametrize("queue_type", [FIFOQueue, FairQueue])
def test_empty_workload_has_zero_throughput(queue_type):
    assert simulate([], queue_type()) == {
        "requests": 0, "simulated_seconds": 0,
        "throughput_requests_per_second": 0, "tenants": {},
    }


@pytest.mark.parametrize("field", ["workers", "tokens_per_second"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_worker_config(field, value):
    with pytest.raises(ValueError, match=field):
        simulate([], FIFOQueue(), **{field: value})


@pytest.mark.parametrize("queue_type", [FIFOQueue, FairQueue])
def test_simulation_rejects_existing_work(queue_type):
    queue = queue_type()
    expected = queue.enqueue("a", "original", cost=100)
    with pytest.raises(ValueError, match="empty queue"):
        simulate([], queue)
    assert queue.dispatch() is expected


def run_module(module, *args):
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, timeout=10,
    )


def test_benchmark_matches_source_workload_and_demonstrates_burst_isolation():
    report = run_comparison()
    assert "simulated workload" in report["mode"]
    assert report["workers"] == 4
    expected = {
        "fifo": [(15650, 29700), (31600, 31900), (32200, 32500), (32850, 33150)],
        "fair": [(17500, 31600), (1150, 2300), (1150, 2300), (1150, 2300)],
    }
    for mode in ("fifo", "fair"):
        result = report[mode]
        assert result["requests"] == 1060
        assert result["simulated_seconds"] == 33.2
        assert result["throughput_requests_per_second"] == 31.928
        for index, (tenant, stats) in enumerate(result["tenants"].items()):
            assert stats["requests"] == (1000 if tenant == "noisy" else 20)
            assert (stats["p50_latency_ms"], stats["p95_latency_ms"]) == expected[mode][index]
    for tenant in ("small-1", "small-2", "small-3"):
        assert report["fair"]["tenants"][tenant]["p95_latency_ms"] < (
            report["fifo"]["tenants"][tenant]["p95_latency_ms"] / 10
        )


def test_benchmark_json_cli_is_deterministic_and_matches_report():
    result = run_module("fairq.bench", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == run_comparison()
    assert result.stdout == run_module("fairq.bench", "--json").stdout


def test_benchmark_table_cli():
    result = run_module("fairq.bench")
    assert result.returncode == 0, result.stderr
    assert "Simulated workload:" in result.stdout
    assert "FIFO p50/p95 (ms)" in result.stdout
    assert "Fair p50/p95 (ms)" in result.stdout
    assert result.stdout.count("1,060 completed") == 2
    assert result.stdout.count("31.928 requests/s") == 2


def test_benchmark_custom_worker_count():
    result = run_module("fairq.bench", "--workers", "1", "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workers"] == 1
    for mode in ("fifo", "fair"):
        assert report[mode]["requests"] == 1060
        assert report[mode]["simulated_seconds"] == 132.5
        assert report[mode]["throughput_requests_per_second"] == 8


@pytest.mark.parametrize("args", [("--workers", "0"), ("--workers", "-1"), ("--workers", "oops")])
def test_benchmark_cli_rejects_invalid_workers(args):
    result = run_module("fairq.bench", *args)
    assert result.returncode == 2
    assert "--workers" in result.stderr
    assert not result.stdout


def test_demo_reports_fifo_fair_order_and_token_weights():
    result = run_module("fairq.demo")
    assert result.returncode == 0, result.stderr
    assert "FIFOQueue: noisy-0 noisy-1 noisy-2" in result.stdout
    assert "FairQueue: noisy-0 small-0 noisy-1 small-1" in result.stdout
    assert "a: 40 requests, 2000 estimated tokens" in result.stdout
    assert "b: 10 requests, 1000 estimated tokens" in result.stdout
