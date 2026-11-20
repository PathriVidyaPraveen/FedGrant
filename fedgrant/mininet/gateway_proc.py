"""Gateway process entry point used inside the Mininet gateway host."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fedgrant.core.state import Status
from fedgrant.mininet.contract import (
    DEFAULT_MODELS_DIR, STUB_POLICY, MininetGatewayContext, build_gateway, load_bank, load_factory,
    policy_config_for, uniform_weights,
)
from fedgrant.mininet.logging_tap import LoggingUdpGateway
from fedgrant.mininet.round_loop import RoundTiming, RoundWindow, readiness_delays_us, run_rounds
from fedgrant.mininet.round_state import RoundStateFile


def build_context(args: argparse.Namespace) -> MininetGatewayContext:
    """Build the factory input; weights come from the value bank when one is given."""
    weights = (load_bank(args.value_bank, args.clients).client_weights
               if args.value_bank else uniform_weights(args.clients))
    return MininetGatewayContext(
        num_clients=args.clients, policy=args.policy, seed=args.seed,
        policy_config=policy_config_for(args.policy), client_weights=weights,
        deadline_ns=args.duration_ns, K=args.blocks, L=args.block_length,
        models_dir=args.models_dir, log_dir=args.log_dir,
    )


def _round_probe_grants(core):
    """Probe grants issued this round (GatewayCore.round_probe_grants); blank without probe support."""
    value = getattr(core, "round_probe_grants", None)
    if callable(value):
        value = value()
    return "" if value is None else int(value)


class NetworkOnlyAggregateSink:
    """Aggregate receiver for network-only runs (M1, smoke): records each aggregate's size.

    M1 measures the network only, so the aggregate G is not applied to a model;
    without a receiver UdpGateway logs an error and counts aggregate_transfer_errors
    every round. One line per round goes to aggregates.jsonl. M2 (FL over Mininet)
    needs a real sender to the FL server instead.
    """

    def __init__(self, log_dir: str | Path):
        self.path = Path(log_dir) / "aggregates.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    async def send_aggregate(self, round_id, aggregate, client_n_coords) -> None:
        record = {"round_id": int(round_id), "dimension": int(getattr(aggregate, "size", 0) or 0),
                  "clients": len(client_n_coords),
                  "delivered_coords": int(sum(sum(blocks.values()) for blocks in client_n_coords.values()))}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


class GrantRoundProtocol:
    """GatewayCore (grant mode) steps for the shared round loop."""

    def __init__(self, core, gateway, *, num_clients: int, K: int, L: int, seed: int = 0):
        self.core, self.gateway = core, gateway
        self.num_clients, self.K, self.L, self.seed = num_clients, K, L, seed
        self._before: dict[str, int] = {}
        self._deadline_completion = 0.0

    def begin_round(self, round_id: int, now_ns: int, duration_ns: int) -> RoundWindow:
        stats = self.core.stats
        self._before = {name: getattr(stats, name) for name in (
            "late_after_timeout", "late_after_deadline", "accepted_data", "rejected_data", "invalid_events")}
        self._fallbacks_before = self._fallbacks()
        # Per-client readiness rho_i ~ U(0, 0.1 D) from the simulator's seeded
        # stream (Section 3.1), the same draws as the M-sim twin and the B1-B3
        # processes; each client waits rho_i after ROUND_START before READY.
        rho_us = readiness_delays_us(self.seed, round_id, self.num_clients, duration_ns)
        self._actions = self.core.on_round_start(now_ns, round_id=round_id, duration_ns=duration_ns,
                                                 K=self.K, L=self.L, rho_us=rho_us)
        return RoundWindow(round_id, self.core.round.start_ns, self.core.round.deadline_ns)

    def _fallbacks(self) -> int | None:
        predictor = getattr(self.core, "predictor", None)
        count = getattr(predictor, "n_fallbacks", None)
        return None if count is None else int(count)

    def send_round_start(self, window: RoundWindow) -> None:
        # The round loop has already published the round window, so client
        # rate replay is relative to the real round origin.
        self.gateway.dispatch_actions(self._actions)
        self._actions = []

    def at_deadline(self, window: RoundWindow) -> None:
        if self.core.phase.name == "ACTIVE":
            # Snapshot before on_deadline marks pending grants lost.
            self.gateway.capture_deadline_state()
            self.gateway.dispatch_actions(self.core.on_deadline(self.core.round.deadline_ns))
        # The transport scheduler may have invoked on_deadline first; its tick
        # hook snapshots unresolved grants before cleanup.
        self._deadline_completion = float(self.gateway.capture_deadline_state() == 0)
        self.gateway.log_timeouts()

    def finish_round(self, window: RoundWindow) -> tuple[dict, dict]:
        # Called after the late drain: arrivals after D were counted against this round.
        core, stats = self.core, self.core.stats
        states = list(core.clients.values())
        total_value = sum(float(state.val.sum(dtype="float64")) for state in states)
        delivered_value = sum(float(state.val[state.status == Status.DELIVERED].sum(dtype="float64"))
                              for state in states)
        eta = delivered_value / total_value if total_value > 0 and len(core.energies) == self.num_clients else ""
        delivered = sum(state.m for state in states)
        issued = max(0, core.next_grant_id - 1)
        timed_out = sum(len(ids) for ids in core.timed_out_grants.values())
        duration_s = (window.deadline_ns - window.start_ns) / 1e9
        row = {
            "eta": eta,
            "delivered_blocks": delivered,
            "clients_represented": sum(state.m > 0 for state in states),
            "deadline_completion": self._deadline_completion,
            "goodput_bps": delivered * 2 * self.L * 8 / duration_s,
            "wasted_grants": timed_out / issued if issued else 0.0,
            "late_after_timeout": stats.late_after_timeout - self._before["late_after_timeout"],
            "late_after_deadline": stats.late_after_deadline - self._before["late_after_deadline"],
            "probe_grants": _round_probe_grants(core),
        }
        record = {name: getattr(stats, name) - self._before[name]
                  for name in ("accepted_data", "rejected_data", "invalid_events")}
        # Section 9 fallback: >0 means the configured predictor (e.g. P-GRU) was
        # replaced by P-EWMA for part of this round; such runs are flagged.
        now = self._fallbacks()
        record["predictor_fallbacks"] = None if now is None else now - self._fallbacks_before
        return row, record


async def serve(args: argparse.Namespace) -> None:
    core = build_gateway(args.gateway_factory, build_context(args))
    endpoints = {i: (f"10.0.0.{i + 1}", args.client_port) for i in range(args.clients)}
    gateway = await LoggingUdpGateway.bind(core, local_addr=("10.0.0.254", args.port),
                                           endpoints=endpoints, log_dir=args.log_dir,
                                           profile_scheduler=args.profile_gateway,
                                           aggregate_sender=aggregate_sender_from_args(args))
    try:
        await run_rounds(
            GrantRoundProtocol(core, gateway, num_clients=args.clients, K=args.blocks, L=args.block_length,
                               seed=args.seed),
            timing_from_args(args), round_state=RoundStateFile(args.round_state), log_dir=args.log_dir)
        # M2: the last round's aggregate must reach the FL server before exit.
        await gateway.flush_aggregates(getattr(args, "aggregate_timeout_s", 10.0) * 2)
    finally:
        await gateway.close()


def parse_endpoint(text: str) -> tuple[str, int]:
    """HOST:PORT -> (host, port)."""
    host, sep, port = str(text).rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) <= 0xFFFF:
        raise ValueError(f"endpoint must be HOST:PORT, got {text!r}")
    return host, int(port)


def aggregate_sender_from_args(args: argparse.Namespace):
    """M2 (FL over Mininet): send each aggregate to the FL server; otherwise network-only."""
    endpoint = getattr(args, "aggregate_endpoint", None)
    if not endpoint:
        return NetworkOnlyAggregateSink(args.log_dir)
    from fedgrant.net.fl_tcp import TcpAggregateSender

    host, port = parse_endpoint(endpoint)
    return TcpAggregateSender(host, port, timeout=getattr(args, "aggregate_timeout_s", 10.0))


def timing_from_args(args: argparse.Namespace) -> RoundTiming:
    return RoundTiming(rounds=args.rounds, duration_ns=args.duration_ns,
                       compute_window_s=args.compute_window_s, late_drain_s=args.late_drain_s,
                       inter_round_s=args.inter_round_s, start_delay_s=args.start_delay_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--duration-ns", type=int, default=400_000_000)
    parser.add_argument("--blocks", type=int, default=169)
    parser.add_argument("--block-length", type=int, default=700)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--client-port", type=int, default=9001)
    parser.add_argument("--start-delay-s", type=float, default=0.4)
    parser.add_argument("--compute-window-s", type=float, default=1.0)
    parser.add_argument("--late-drain-s", type=float, default=0.1)
    parser.add_argument("--inter-round-s", type=float, default=0.0)
    parser.add_argument("--gateway-factory", default="fedgrant.mininet.stub_policy:create_gateway")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--round-state", required=True)
    parser.add_argument("--profile-gateway", action="store_true")
    parser.add_argument("--policy", default=STUB_POLICY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-bank", default=None,
                        help="FULL value bank (.npz); client weights come from it. Default: uniform 1/N")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--aggregate-endpoint", default=None,
                        help="M2: FL server HOST:PORT that receives each round's aggregate "
                             "(default: network-only sink, aggregates.jsonl)")
    parser.add_argument("--aggregate-timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)
    if (args.clients < 1 or args.rounds < 1 or args.duration_ns < 1
            or args.compute_window_s < 0 or args.late_drain_s < 0 or args.inter_round_s < 0):
        parser.error("clients, rounds, and duration must be positive; wait intervals must be non-negative")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
