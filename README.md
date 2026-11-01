# fairq

A small, in-memory queue for shared workers. A noisy tenant can dump a thousand
requests into it without making everyone else wait for that whole burst.
Request cost is estimated tokens, and each tenant keeps its own FIFO queue.
There's also a global `FIFOQueue` for comparison. No runtime dependencies,
services, API keys, or paid calls.

## Quickstart

You'll need [uv](https://docs.astral.sh/uv/). From a fresh clone, run these in order.
uv installs Python 3.12 if needed and includes pytest in the default sync.

```bash
uv sync
uv run python -m fairq.demo
uv run python -m fairq.bench
uv run pytest
```

The demo prints FIFO and fair dispatch order, then shows a 2:1 token share.
The benchmark runs a simulated workload locally and prints the table below.
For the full report as JSON, run `uv run python -m fairq.bench --json`.

## Use it

```python
from fairq import FairQueue

queue = FairQueue[str](quantum=100)
queue.enqueue("busy", "long prompt", cost=200, weight=2)
queue.enqueue("small", "short prompt", cost=50)

while queue:
    request = queue.dispatch()
    print(request.tenant, request.item, request.cost)
```

`enqueue(tenant, item, *, cost, weight=None)` returns an immutable `Request`
holding your payload. `dispatch()` removes one request, or raises `IndexError`
when empty. `len(queue)` counts queued requests. Swap in `FIFOQueue` for global
arrival order; it accepts the same enqueue arguments and ignores weight.

`FairQueue` uses weighted deficit round robin. Each visit adds `quantum * weight`
tokens, then spends that credit on whole requests. Remaining credit carries
between rounds, so a request bigger than the quantum eventually runs. Idle
tenants lose credit and rejoin at the back when new work arrives.

Weights default to 1. Set them on enqueue or with
`FairQueue(weights={"busy": 2})`. A weight stays set when a tenant goes idle.
You can change it while that tenant has no queued work; changing it with work
queued raises `ValueError`. Cost, quantum, and weights must be positive integers
(not booleans), and tenant names must be nonempty strings. Invalid input raises
`ValueError` without adding work.

Use one dispatcher per queue. Call `dispatch()` when a worker is free; the queue
doesn't run work or preempt it. Fairness is about estimated token service among
backlogged tenants, so inaccurate estimates and long running requests affect
latency. The queue has no locking or persistence.

## A measured simulation

One noisy tenant queues 1,000 requests first, then three small tenants queue 20
each. All arrive at simulated time zero. Costs cycle through 50/100/150/200 tokens.
Four simulated workers each process 1,000 tokens per second. Quantum is 100 and
all weights are 1.

Actual output from `uv run python -m fairq.bench`:

| Tenant | Requests | FIFO p50 / p95 (ms) | Fair p50 / p95 (ms) |
| --- | --- | --- | --- |
| noisy | 1,000 | 15,650 / 29,700 | 17,500 / 31,600 |
| small-1 | 20 | 31,600 / 31,900 | 1,150 / 2,300 |
| small-2 | 20 | 32,200 / 32,500 | 1,150 / 2,300 |
| small-3 | 20 | 32,850 / 33,150 | 1,150 / 2,300 |

FIFO and fair each finish **1,060 requests in 33.2 simulated seconds**, for
**31.928 requests/second**. FIFO leaves the small tenants waiting behind the
burst. Fair service gets them through sooner at the noisy tenant's expense.
This finite burst shows temporary starvation, not an infinite wait.

These are deterministic simulated completion latencies, including queue wait
and service, with nearest-rank percentiles. The numbers come from dispatching
every request through the queues. They aren't wall-clock performance claims.
Scheduling overhead, batching, and estimation error aren't modeled.

## Credit

Extracted from infergate's MIT-licensed `scheduling.py`, scheduler tests, and
`fair_loadtest.py` at commit `89359e91078f6f314899ecf7219f769f117f5473`.
Original copyright: Truman Chan, 2026. Inspired by
[Cohere's serving fairness post](https://cohere.com/blog/serving-fairness).
fairq implements per-tenant FIFO and weighted deficit round robin. MIT licensed.
