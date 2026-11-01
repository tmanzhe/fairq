"""A tiny burst showing dispatch order and weighted token shares."""

from . import FIFOQueue, FairQueue


def main() -> None:
    print("Simulated burst: 8 noisy requests, then 2 small; cost=100 tokens each.")
    for queue in (FIFOQueue[str](), FairQueue[str]()):
        for tenant, count in (("noisy", 8), ("small", 2)):
            for index in range(count):
                queue.enqueue(tenant, f"{tenant}-{index}", cost=100)
        order = []
        while queue:
            order.append(queue.dispatch().item)
        print(f"{type(queue).__name__}: {' '.join(order)}")

    queue = FairQueue[str](quantum=100)
    for tenant, cost, weight in (("a", 50, 2), ("b", 100, 1)):
        for index in range(100):
            queue.enqueue(tenant, f"{tenant}-{index}", cost=cost, weight=weight)
    tokens = {"a": 0, "b": 0}
    counts = {"a": 0, "b": 0}
    for _ in range(50):
        request = queue.dispatch()
        tokens[request.tenant] += request.cost
        counts[request.tenant] += 1
    print("First 50 weighted dispatches (a weight=2, b weight=1):")
    for tenant in tokens:
        print(f"  {tenant}: {counts[tenant]} requests, {tokens[tenant]} estimated tokens")


if __name__ == "__main__":
    main()
