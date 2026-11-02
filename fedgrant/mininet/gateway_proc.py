"""Gateway process entry point used inside the Mininet gateway host."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import csv
import json
import time
from pathlib import Path

from fedgrant.common.logschema import ROUND_METRIC_COLUMNS
from fedgrant.core.state import Status
from fedgrant.mininet.logging_tap import LoggingUdpGateway
from fedgrant.mininet.round_state import RoundStateFile


def load_factory(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("gateway factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), attribute)


async def serve(args: argparse.Namespace) -> None:
    core = load_factory(args.gateway_factory)(args.clients)
    endpoints = {i: (f"10.0.0.{i + 1}", args.client_port) for i in range(args.clients)}
    gateway = await LoggingUdpGateway.bind(core, local_addr=("10.0.0.254", args.port),
                                           endpoints=endpoints, log_dir=args.log_dir)
    try:
        round_state = RoundStateFile(args.round_state)
        round_state.write(round_id=0, active=False)
        await asyncio.sleep(args.start_delay_s)
        round_metrics_path = Path(args.log_dir, "round_metrics.csv")
        round_records = []
        with round_metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=ROUND_METRIC_COLUMNS, extrasaction="raise")
            metrics_writer.writeheader()
            for round_id in range(1, args.rounds + 1):
                await asyncio.sleep(args.compute_window_s)
                late_timeout_before = core.stats.late_after_timeout
                late_deadline_before = core.stats.late_after_deadline
                accepted_before = core.stats.accepted_data
                rejected_before = core.stats.rejected_data
                invalid_before = core.stats.invalid_events
                start_ns = time.monotonic_ns()
                actions = core.on_round_start(start_ns, round_id=round_id,
                                              duration_ns=args.duration_ns,
                                              K=args.blocks, L=args.block_length)
                round_start_ns = core.round.start_ns
                round_deadline_ns = core.round.deadline_ns
                round_state.write(round_id=round_id, start_ns=round_start_ns,
                                  deadline_ns=round_deadline_ns, active=True)
                # Publish the trace/telemetry window before the first UDP
                # ROUND_START is sent, so every client rate replay is relative
                # to the actual round origin.
                gateway.dispatch_actions(actions)
                await asyncio.sleep(args.duration_ns / 1e9)
                if core.phase.name == "ACTIVE":
                    # Snapshot before on_deadline marks pending grants lost.
                    gateway.capture_deadline_state()
                    gateway.dispatch_actions(core.on_deadline(core.round.deadline_ns))
                # The transport scheduler may have invoked on_deadline first;
                # its tick hook snapshots unresolved grants before cleanup.
                unresolved_at_deadline = gateway.capture_deadline_state()
                deadline_completion = float(unresolved_at_deadline == 0)
                gateway.log_timeouts()
                round_state.write(round_id=round_id, active=False)
                # Keep the old round context available while queued DATA
                # packets drain, so arrivals after D are counted against this
                # round before the next ROUND_START resets the counters.
                await asyncio.sleep(args.late_drain_s)
                states = list(core.clients.values())
                total_value = sum(float(state.val.sum(dtype="float64")) for state in states)
                delivered_value = sum(float(state.val[state.status == Status.DELIVERED].sum(dtype="float64")) for state in states)
                eta = delivered_value / total_value if total_value > 0 and len(core.energies) == args.clients else ""
                delivered = sum(state.m for state in states)
                delivered_payload_bytes = delivered * 2 * args.block_length
                issued = max(0, core.next_grant_id - 1)
                timed_out = sum(len(ids) for ids in core.timed_out_grants.values())
                metrics_writer.writerow({
                    "round_id": round_id,
                    "round_start_mono_ns": round_start_ns,
                    "deadline_mono_ns": round_deadline_ns,
                    "eta": eta,
                    "delivered_blocks": delivered,
                    "clients_represented": sum(state.m > 0 for state in states),
                    "deadline_completion": deadline_completion,
                    "goodput_bps": delivered_payload_bytes * 8 / (args.duration_ns / 1e9),
                    "wasted_grants": timed_out / issued if issued else 0.0,
                    "late_after_timeout": core.stats.late_after_timeout - late_timeout_before,
                    "late_after_deadline": core.stats.late_after_deadline - late_deadline_before,
                })
                metrics_file.flush()
                round_records.append({"round_id": round_id, "start_mono_ns": round_start_ns,
                                      "deadline_mono_ns": round_deadline_ns,
                                      "accepted_data": core.stats.accepted_data - accepted_before,
                                      "rejected_data": core.stats.rejected_data - rejected_before,
                                      "invalid_events": core.stats.invalid_events - invalid_before})
                if round_id < args.rounds:
                    await asyncio.sleep(args.inter_round_s)
        Path(args.log_dir, "gateway_summary.json").write_text(json.dumps(round_records, indent=2) + "\n")
    finally:
        await gateway.close()


def main() -> None:
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
    args = parser.parse_args()
    if (args.clients < 1 or args.rounds < 1 or args.duration_ns < 1
            or args.compute_window_s < 0 or args.late_drain_s < 0 or args.inter_round_s < 0):
        parser.error("clients, rounds, and duration must be positive; wait intervals must be non-negative")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
