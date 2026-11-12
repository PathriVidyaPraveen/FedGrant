"""Gateway process entry point used inside the Mininet gateway host."""

from __future__ import annotations

import argparse
import asyncio

from fedgrant.core.state import Status
from fedgrant.mininet.contract import (
    DEFAULT_MODELS_DIR, STUB_POLICY, MininetGatewayContext, build_gateway, load_bank, load_factory,
    policy_config_for, uniform_weights,
)
from fedgrant.mininet.logging_tap import LoggingUdpGateway
from fedgrant.mininet.round_loop import RoundTiming, RoundWindow, run_rounds
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


class GrantRoundProtocol:
    """GatewayCore (grant mode) steps for the shared round loop."""

    def __init__(self, core, gateway, *, num_clients: int, K: int, L: int):
        self.core, self.gateway = core, gateway
        self.num_clients, self.K, self.L = num_clients, K, L
        self._before: dict[str, int] = {}
        self._deadline_completion = 0.0

    def begin_round(self, round_id: int, now_ns: int, duration_ns: int) -> RoundWindow:
        stats = self.core.stats
        self._before = {name: getattr(stats, name) for name in (
            "late_after_timeout", "late_after_deadline", "accepted_data", "rejected_data", "invalid_events")}
        self._actions = self.core.on_round_start(now_ns, round_id=round_id, duration_ns=duration_ns,
                                                 K=self.K, L=self.L)
        return RoundWindow(round_id, self.core.round.start_ns, self.core.round.deadline_ns)

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
        }
        record = {name: getattr(stats, name) - self._before[name]
                  for name in ("accepted_data", "rejected_data", "invalid_events")}
        return row, record


async def serve(args: argparse.Namespace) -> None:
    core = build_gateway(args.gateway_factory, build_context(args))
    endpoints = {i: (f"10.0.0.{i + 1}", args.client_port) for i in range(args.clients)}
    gateway = await LoggingUdpGateway.bind(core, local_addr=("10.0.0.254", args.port),
                                           endpoints=endpoints, log_dir=args.log_dir,
                                           profile_scheduler=args.profile_gateway)
    try:
        await run_rounds(
            GrantRoundProtocol(core, gateway, num_clients=args.clients, K=args.blocks, L=args.block_length),
            timing_from_args(args), round_state=RoundStateFile(args.round_state), log_dir=args.log_dir)
    finally:
        await gateway.close()


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
    args = parser.parse_args(argv)
    if (args.clients < 1 or args.rounds < 1 or args.duration_ns < 1
            or args.compute_window_s < 0 or args.late_drain_s < 0 or args.inter_round_s < 0):
        parser.error("clients, rounds, and duration must be positive; wait intervals must be non-negative")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
