"""Full-participation federated averaging and periodic evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from fedgrant.encoder.blocks import encode
from fedgrant.encoder.ef import ErrorFeedback
from fedgrant.fl.bank import ValueBank
from fedgrant.fl.local import CyclicBatchLoader, train_client
from fedgrant.fl.model import FEMNISTCNN, get_flat_params, set_flat_params
from fedgrant.policy.constants import D_MODEL, K


from fedgrant.policy.types import PolicyConfig
from fedgrant.sim.run_baselines import BaselineState, run_baseline_round
from fedgrant.sim.run_net import run_network_round
from fedgrant.sim.seeding import network_round_seed as _network_round_seed



def aggregate_updates(updates: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean of decoded client updates in float32."""
    w = np.asarray(weights, dtype=np.float64)
    if len(updates) == 0 or w.shape != (len(updates),):
        raise ValueError("one weight is required for every client update")
    if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be finite, non-negative, and sum to 1")
    if any(not isinstance(u, np.ndarray) or u.dtype != np.float32 or u.shape != (D_MODEL,) for u in updates):
        raise ValueError(f"every update must be a float32 vector of shape ({D_MODEL},)")
    out = np.zeros(D_MODEL, dtype=np.float64)
    for wi, update in zip(w, updates, strict=True):
        out += wi * update.astype(np.float64)
    return out.astype(np.float32)


@torch.inference_mode()
def evaluate(
    model: FEMNISTCNN,
    test_clients: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> tuple[float, float]:
    """Evaluate on all client test shards, returning sample-weighted loss and accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    dev = torch.device(device)
    model.to(dev)
    for images, labels in test_clients:
        for start in range(0, len(labels), batch_size):
            x = images[start : start + batch_size].to(dev)
            y = labels[start : start + batch_size].to(dev)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += len(y)
    if total_count == 0:
        raise ValueError("FEMNIST test split is empty")
    return total_loss / total_count, total_correct / total_count


def run_full(
    model: FEMNISTCNN,
    train_clients: Sequence[tuple[torch.Tensor, torch.Tensor]],
    test_client_loader: Callable[[], Sequence[tuple[torch.Tensor, torch.Tensor]]],
    client_weights: np.ndarray,
    loaders: Sequence[CyclicBatchLoader],
    error_feedback: Sequence[ErrorFeedback],
    num_rounds: int = 300,
    seed: int = 0,
    device: torch.device | str = "cpu",
    eval_every: int = 5,
    bank: ValueBank | None = None,
    log_every: int = 5,
    client_workers: int = 10,
) -> list[dict[str, float | int]]:
    """Train all clients each round; encode u=delta+e and decode all fp16 blocks."""
    n = len(train_clients)
    weights = np.asarray(client_weights, dtype=np.float32)
    if n == 0 or len(loaders) != n or len(error_feedback) != n or weights.shape != (n,):
        raise ValueError("client data, loaders, EF states, and weights must have matching lengths")
    if not np.isclose(weights.sum(dtype=np.float64), 1.0):
        raise ValueError("client weights must sum to 1")
    if bank is not None and not np.array_equal(bank.client_weights, weights):
        raise ValueError("bank weights must match training client weights")

    dev = torch.device(device)
    model.to(dev)
    history: list[dict[str, float | int]] = []
    if client_workers < 1:
        raise ValueError("client_workers must be >= 1")

    def train_one(client_id: int, round_id: int) -> np.ndarray:
        images, labels = train_clients[client_id]
        return train_client(
            client_id, images, labels, model, round_idx=round_id - 1,
            master_seed=seed, loader=loaders[client_id], device=dev,
        )

    with ThreadPoolExecutor(max_workers=min(client_workers, n)) as executor:
        for round_id in range(1, num_rounds + 1):
            # All local jobs read the same immutable round-start model. executor.map
            # returns results in client-id order, so aggregation stays deterministic.
            deltas = list(executor.map(lambda i: train_one(i, round_id), range(n)))
            decoded: list[np.ndarray] = []
            energy_rows: list[np.ndarray] = []
            for delta, ef in zip(deltas, error_feedback, strict=True):
                u = ef.prepare(delta)
                encoded = encode(u)
                # FULL delivers every block. Decoding still applies fp16 rounding,
                # and finish_round stores the resulting rounding residual.
                decoded.append(ef.finish_round(u, np.arange(encoded.K, dtype=np.int64)))
                energy_rows.append(encoded.energies_sorted)

            if bank is not None:
                bank.add_round(round_id, np.stack(energy_rows))
            theta = get_flat_params(model)
            theta += aggregate_updates(decoded, weights)
            set_flat_params(model, theta)

            if round_id % eval_every == 0 or round_id == num_rounds:
                test_clients = test_client_loader()
                loss, accuracy = evaluate(model, test_clients, dev)
                metrics: dict[str, float | int] = {
                    "round": round_id,
                    "test_loss": loss,
                    "test_accuracy": accuracy,
                }
                history.append(metrics)
                if round_id % log_every == 0 or round_id == num_rounds:
                    print(
                        f"seed={seed} round={round_id}/{num_rounds} "
                        f"test_loss={loss:.5f} test_accuracy={accuracy:.4f}",
                        flush=True,
                    )
    return history


def _plan_data_rtt_send_pos(round_result, encoded_updates, client_ids) -> list[int] | None:
    """send_pos of the DATA that produced each B3 RTT sample (parallel to client_ids).

    The sample is taken at a client's first accepted DATA, which is the first
    entry of the accumulator's ``receive_order`` (accepted blocks in arrival
    order); B3 sends block ``order[send_pos - 1]`` at ``send_pos``. Returns
    None, rather than inventing values, when that information is unavailable.
    """
    results = getattr(round_result, "client_results", None)
    if results is None:
        return None
    positions: list[int] = []
    for cid in client_ids:
        order = getattr(results.get(cid), "receive_order", None)
        if not order:
            return None
        matches = np.flatnonzero(np.asarray(encoded_updates[cid].order) == int(order[0]))
        if matches.size != 1:
            return None
        positions.append(int(matches[0]) + 1)
    return positions


def run_fl_simulated(
    model: FEMNISTCNN,
    train_clients: Sequence[tuple[torch.Tensor, torch.Tensor]],
    test_client_loader: Callable[[], Sequence[tuple[torch.Tensor, torch.Tensor]]],
    client_weights: np.ndarray,
    loaders: Sequence[CyclicBatchLoader],
    error_feedback: Sequence[ErrorFeedback],
    num_rounds: int = 300,
    seed: int = 0,
    beta: float = 0.10,
    scenario: str = "L2",
    split: str = "test",
    trace_dir: str = "data/traces",
    policy_config: PolicyConfig | None = None,
    transport_baseline: str | None = None,
    device: torch.device | str = "cpu",
    eval_every: int = 5,
    log_every: int = 5,
    client_workers: int = 10,
) -> list[dict[str, Any]]:
    """Execute end-to-end FL training over the simulated network (Sections 5.1, 6, 11).

    In each round:
    1. Clients perform 20 local SGD steps to produce delta_i.
    2. Clients compute u_i = delta_i + e_i (error feedback) and encode into energy-ordered blocks.
    3. The network simulator simulates frame transmission over the lossy, bandwidth-constrained network.
    4. Delivered blocks are decoded and error feedback e_i <- u_i - u_hat_i is updated.
    5. Server aggregates delivered updates: x_{r+1} = x_r + sum_i w_i u_hat_i.
    """
    n = len(train_clients)
    weights = np.asarray(client_weights, dtype=np.float32)
    if n == 0 or len(loaders) != n or len(error_feedback) != n or weights.shape != (n,):
        raise ValueError("client data, loaders, EF states, and weights must have matching lengths")
    if not np.isclose(weights.sum(dtype=np.float64), 1.0):
        raise ValueError("client weights must sum to 1")
    if transport_baseline is not None:
        transport_baseline = transport_baseline.upper()
        if transport_baseline not in {"B2", "B3"}:
            raise ValueError("transport_baseline must be B2 or B3")

    dev = torch.device(device)
    model.to(dev)
    history: list[dict[str, Any]] = []
    gateway = None
    baseline_state: BaselineState | None = None

    def train_one(client_id: int, round_id: int) -> np.ndarray:
        images, labels = train_clients[client_id]
        return train_client(
            client_id, images, labels, model, round_idx=round_id - 1,
            master_seed=seed, loader=loaders[client_id], device=dev,
        )

    with ThreadPoolExecutor(max_workers=min(client_workers, n)) as executor:
        for round_id in range(1, num_rounds + 1):
            # 1. Local training on clients
            deltas = list(executor.map(lambda i: train_one(i, round_id), range(n)))

            # 2. Error feedback and block encoding
            u_list = [ef.prepare(delta) for delta, ef in zip(deltas, error_feedback, strict=True)]
            enc_list = [encode(u) for u in u_list]

            energies_sorted = np.stack([enc.energies_sorted for enc in enc_list])
            send_orders = [enc.order for enc in enc_list]
            client_payloads = [{j: enc.blocks[j] for j in range(enc.K)} for enc in enc_list]

            # 3. Simulate network round
            D_ns = 398_000_000  # Default D_ns for beta=0.10
            t_full_s = 3.985 * (n / 20.0) * (K / 169.0)
            d_ms = int(round(beta * t_full_s * 1000.0))
            D_ns = d_ms * 1_000_000
            round_start_abs_ns = int((round_id - 1) * (1_000_000_000 + D_ns))

            network_seed = _network_round_seed(seed, round_id)
            baseline_result = None
            if transport_baseline is None:
                net_res = run_network_round(
                    round_id=round_id,
                    num_clients=n,
                    client_weights=weights,
                    client_energies=energies_sorted,
                    client_payloads=client_payloads,
                    send_orders=send_orders,
                    beta=beta,
                    scenario=scenario,
                    split=split,
                    trace_dir=trace_dir,
                    policy_config=policy_config,
                    seed=network_seed,
                    gateway=gateway,
                    round_start_abs_ns=round_start_abs_ns,
                )
                gateway = net_res["gateway"]  # Persist gateway state across rounds
                delivered_indices = [re.delivered_indices for re in net_res["round_ends"]]
            else:
                baseline_result = run_baseline_round(
                    transport_baseline,
                    round_id=round_id,
                    num_clients=n,
                    client_weights=weights,
                    encoded_updates=enc_list,
                    beta=beta,
                    scenario=scenario,
                    split=split,
                    trace_dir=trace_dir,
                    seed=network_seed,
                    state=baseline_state,
                    round_start_abs_ns=round_start_abs_ns,
                )
                baseline_state = baseline_result["baseline_state"]
                round_result = baseline_result["round_result"]
                delivered_indices = [
                    round_result.client_results[cid].delivered_indices for cid in range(n)
                ]
                net_res = {
                    "eta": baseline_result["eta"],
                    "delivered_counts": baseline_result["delivered_counts"],
                    "metrics": baseline_result["metrics"],
                    "late_metrics": baseline_result["late_metrics"],
                }

            # 4. Client error feedback update based on delivered ROUND_END bitmap
            decoded = [
                ef.finish_round(u, indices)
                for ef, u, indices in zip(error_feedback, u_list, delivered_indices, strict=True)
            ]

            # 5. Global model aggregation
            theta = get_flat_params(model)
            theta += aggregate_updates(decoded, weights)
            set_flat_params(model, theta)

            # 6. Periodic evaluation
            record: dict[str, Any] = {
                "round": round_id,
                "eta": net_res["eta"],
                "delivered_blocks": sum(net_res["delivered_counts"]),
                "metrics": net_res["metrics"],
                "late_metrics": net_res["late_metrics"],
                "rtt_samples_ns": (
                    [] if baseline_result is not None else net_res.get("rtt_samples_ns", [])
                ),
            }
            if transport_baseline == "B3" and baseline_result is not None:
                # B3 mode-2 RTT (Section 10): first accepted DATA arrival at the
                # gateway minus that client's PLAN send time, at most one sample
                # per client per round, produced by B3GatewayCore. Kept separate
                # from grant RTT. The two lists are parallel and client-sorted;
                # clients without a sample (no PLAN handoff / no DATA) are absent.
                b3_rtt = baseline_result.get("b3_rtt_samples_ns", {})
                client_ids = sorted(int(cid) for cid in b3_rtt)
                record["plan_data_rtt_client_ids"] = client_ids
                record["plan_data_rtt_samples_ns"] = [int(b3_rtt[cid]) for cid in client_ids]
                send_pos = _plan_data_rtt_send_pos(baseline_result.get("round_result"), enc_list, client_ids)
                if send_pos is not None:
                    record["plan_data_rtt_send_pos"] = send_pos
            if round_id % eval_every == 0 or round_id == num_rounds:
                test_clients = test_client_loader()
                loss, accuracy = evaluate(model, test_clients, dev)
                record["test_loss"] = loss
                record["test_accuracy"] = accuracy
                if round_id % log_every == 0 or round_id == num_rounds:
                    print(
                        f"seed={seed} round={round_id}/{num_rounds} "
                        f"test_loss={loss:.5f} test_acc={accuracy:.4f} "
                        f"eta={net_res['eta']:.4f} deliv_blocks={sum(net_res['delivered_counts'])}",
                        flush=True,
                    )
            history.append(record)

    return history


__all__ = ["aggregate_updates", "evaluate", "run_full", "run_fl_simulated"]
