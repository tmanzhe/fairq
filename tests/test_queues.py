"""Dispatch behavior adapted and expanded from infergate's scheduler tests."""

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from fairq import FIFOQueue, FairQueue, Request


def enqueue(queue, tenant, costs, **kwargs):
    return [
        queue.enqueue(tenant, f"{tenant}-{index}", cost=cost, **kwargs)
        for index, cost in enumerate(costs)
    ]


def drain(queue):
    result = []
    while queue:
        result.append(queue.dispatch())
    return result


@pytest.mark.parametrize("queue_type", [FIFOQueue, FairQueue])
def test_empty_length_and_reuse(queue_type):
    queue = queue_type()
    assert not queue
    assert len(queue) == 0
    with pytest.raises(IndexError, match="empty"):
        queue.dispatch()
    request = queue.enqueue("a", None, cost=100)
    assert queue and len(queue) == 1
    assert queue.dispatch() is request
    assert not queue
    assert drain(queue) == []
    assert drain_after_enqueue(queue) == ["a-0", "a-1"]


def drain_after_enqueue(queue):
    enqueue(queue, "a", [100, 100])
    return [request.item for request in drain(queue)]


@pytest.mark.parametrize("queue_type", [FIFOQueue, FairQueue])
def test_opaque_payload_is_preserved_and_metadata_is_immutable(queue_type):
    payload = {"prompt": "hello"}
    queue = queue_type()
    request = queue.enqueue("a", payload, cost=42)
    with pytest.raises(FrozenInstanceError):
        request.cost = 1
    dispatched = queue.dispatch()
    assert dispatched is request
    assert dispatched.item is payload
    assert (dispatched.tenant, dispatched.cost) == ("a", 42)


def test_fifo_keeps_global_arrival_order_even_with_weights():
    queue = FIFOQueue()
    expected = enqueue(queue, "noisy", [200] * 1000)
    expected += enqueue(queue, "small", [50] * 20, weight=100)
    assert drain(queue) == expected


def test_fair_isolates_small_tenant_and_preserves_per_tenant_fifo():
    queue = FairQueue()
    noisy = enqueue(queue, "noisy", [100] * 1000)
    small = enqueue(queue, "small", [100] * 20)
    completed = drain(queue)
    assert completed[:4] == [noisy[0], small[0], noisy[1], small[1]]
    assert completed.index(small[-1]) == 39
    for tenant, expected in (("noisy", noisy), ("small", small)):
        assert [request for request in completed if request.tenant == tenant] == expected
    assert len({request.item for request in completed}) == 1020


def test_fifo_burst_starves_small_tenant_until_noisy_backlog_is_dispatched():
    queues = [FIFOQueue(), FairQueue()]
    for queue in queues:
        enqueue(queue, "noisy", [100] * 1000)
        enqueue(queue, "small", [100] * 20)
    fifo_prefix, fair_prefix = ([q.dispatch() for _ in range(40)] for q in queues)
    assert Counter(r.tenant for r in fifo_prefix) == {"noisy": 40}
    assert Counter(r.tenant for r in fair_prefix) == {"noisy": 20, "small": 20}


@pytest.mark.parametrize("via_enqueue", [False, True])
def test_weight_two_allocates_twice_the_tokens(via_enqueue):
    queue = FairQueue(weights={} if via_enqueue else {"a": 2})
    enqueue(queue, "a", [50] * 100, **({"weight": 2} if via_enqueue else {}))
    enqueue(queue, "b", [100] * 100)
    first = [queue.dispatch() for _ in range(50)]
    tokens_a = sum(r.cost for r in first if r.tenant == "a")
    tokens_b = sum(r.cost for r in first if r.tenant == "b")
    assert tokens_a == 2 * tokens_b == 2000


def test_equal_weights_balance_cost_not_request_counts():
    queue = FairQueue(quantum=100)
    enqueue(queue, "cheap", [25] * 100)
    enqueue(queue, "expensive", [100] * 100)
    first = [queue.dispatch() for _ in range(50)]
    assert Counter(r.tenant for r in first) == {"cheap": 40, "expensive": 10}
    assert sum(r.cost for r in first if r.tenant == "cheap") == 1000
    assert sum(r.cost for r in first if r.tenant == "expensive") == 1000


def test_large_head_accumulates_credit_without_reordering_or_starving():
    queue = FairQueue(quantum=100)
    large = enqueue(queue, "large", [350, 10])
    small = enqueue(queue, "small", [100] * 10)
    result = drain(queue)
    assert result[:5] == [*small[:3], *large]
    assert len(result) == 12


def test_unspent_credit_is_carried_between_rounds():
    queue = FairQueue(quantum=100)
    a = enqueue(queue, "a", [60, 140])
    b = enqueue(queue, "b", [100] * 4)
    assert drain(queue) == [a[0], b[0], a[1], *b[1:]]


def test_idle_tenant_drops_credit_and_rejoins_at_end():
    queue = FairQueue(quantum=100)
    enqueue(queue, "a", [10])
    b = enqueue(queue, "b", [100] * 10)
    assert queue.dispatch().tenant == "a"
    a = enqueue(queue, "a", [150])
    # The 90 unspent tokens from the previous active period must be gone.
    assert [queue.dispatch() for _ in range(3)] == [b[0], b[1], a[0]]


def test_new_tenant_joins_during_an_active_visit():
    queue = FairQueue(quantum=100)
    a = enqueue(queue, "a", [50] * 4)
    assert queue.dispatch() == a[0]
    b = enqueue(queue, "b", [100])
    assert drain(queue) == [a[1], b[0], a[2], a[3]]


def test_single_tenant_large_cost_and_length_accounting():
    queue = FairQueue(quantum=1)
    expected = enqueue(queue, "a", [10000, 1, 7])
    for index, request in enumerate(expected):
        assert len(queue) == 3 - index
        assert queue.dispatch() is request
    assert len(queue) == 0


def test_three_continuously_backlogged_tenants_all_receive_service():
    queue = FairQueue(quantum=100)
    for tenant in ("a", "b", "c"):
        queue.enqueue(tenant, 0, cost=100)
    served = []
    for index in range(90):
        request = queue.dispatch()
        served.append(request.tenant)
        queue.enqueue(request.tenant, index + 1, cost=100)
    assert served == ["a", "b", "c"] * 30
    assert len(queue) == 3


def test_weight_mapping_is_copied():
    weights = {"a": 2}
    queue = FairQueue(weights=weights)
    weights["a"] = 100
    a = enqueue(queue, "a", [100] * 4)
    b = enqueue(queue, "b", [100] * 4)
    assert [queue.dispatch() for _ in range(3)] == [a[0], a[1], b[0]]


def test_weight_persists_when_tenant_goes_idle():
    queue = FairQueue()
    queue.enqueue("a", "initial", cost=100, weight=2)
    queue.dispatch()
    a = enqueue(queue, "a", [100] * 4)
    b = enqueue(queue, "b", [100] * 4)
    assert [queue.dispatch() for _ in range(3)] == [a[0], a[1], b[0]]


def test_weight_change_is_allowed_when_idle():
    queue = FairQueue(weights={"a": 10})
    queue.enqueue("a", "first", cost=100)
    queue.dispatch()
    a = enqueue(queue, "a", [100] * 4, weight=1)
    b = enqueue(queue, "b", [100] * 4)
    assert [queue.dispatch() for _ in range(4)] == [a[0], b[0], a[1], b[1]]


def test_active_weight_change_is_rejected_without_changing_work_or_credit():
    queue = FairQueue()
    a = enqueue(queue, "a", [50] * 4, weight=2)
    assert queue.dispatch() == a[0]
    with pytest.raises(ValueError, match="cannot change weight"):
        queue.enqueue("a", "rejected", cost=1, weight=1)
    enqueue(queue, "a", [50], weight=2)
    b = enqueue(queue, "b", [100])
    assert [queue.dispatch() for _ in range(4)] == [*a[1:], b[0]]
    assert len(queue) == 1


INVALID_NUMBERS = [0, -1, 1.5, float("nan"), float("inf"), True, False, "1", None]


@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_invalid_cost_and_quantum_are_rejected(value):
    with pytest.raises(ValueError, match="cost"):
        Request("a", "item", value)
    with pytest.raises(ValueError, match="quantum"):
        FairQueue(quantum=value)
    for queue in (FIFOQueue(), FairQueue()):
        with pytest.raises(ValueError, match="cost"):
            queue.enqueue("a", "item", cost=value)
        assert not queue


@pytest.mark.parametrize("value", INVALID_NUMBERS[:-1])
def test_invalid_weights_are_rejected(value):
    with pytest.raises(ValueError, match="weight"):
        FairQueue(weights={"a": value})
    for queue in (FIFOQueue(), FairQueue()):
        with pytest.raises(ValueError, match="weight"):
            queue.enqueue("a", "item", cost=10, weight=value)
        assert not queue


@pytest.mark.parametrize("tenant", ["", "  ", None, 1, True])
def test_invalid_tenants_are_rejected(tenant):
    with pytest.raises(ValueError, match="tenant"):
        FairQueue(weights={tenant: 2})
    for queue in (FIFOQueue(), FairQueue()):
        with pytest.raises(ValueError, match="tenant"):
            queue.enqueue(tenant, "item", cost=10)
        assert not queue


def test_invalid_enqueue_does_not_change_an_existing_tenants_weight():
    queue = FairQueue(weights={"a": 2})
    with pytest.raises(ValueError):
        queue.enqueue("a", "bad", cost=0, weight=100)
    a = enqueue(queue, "a", [100] * 4)
    b = enqueue(queue, "b", [100])
    assert [queue.dispatch() for _ in range(3)] == [a[0], a[1], b[0]]
