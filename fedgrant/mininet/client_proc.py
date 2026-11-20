"""UDP client process entry point used inside a Mininet client host."""

from __future__ import annotations

import argparse
import csv
import asyncio
import json
from pathlib import Path

import numpy as np

from fedgrant.core.client_core import ClientCore
from fedgrant.mininet.contract import bank_update_provider, load_bank
from fedgrant.mininet.profile_update import profile_bank, synthetic_update
from fedgrant.common.constants import MessageType
from fedgrant.common.logschema import PACKET_COLUMNS
from fedgrant.common.wire import decode_message, WireProtocolError
from fedgrant.net.udp_client import UdpClient


class PrefetchedUpdates:
    """Build round r's update before its ROUND_START (Section 7: clients finish within Tc).

    The ROUND_START handler then only looks the update up, so READY follows
    ROUND_START by rho_i, as in the simulator and the B1-B3 clients (D-20).
    """

    def __init__(self, provider, rounds: int):
        self.provider, self.rounds = provider, int(rounds)
        self._cache: dict[int, np.ndarray] = {}

    def prefetch(self, round_id: int) -> None:
        if 1 <= round_id <= self.rounds and round_id not in self._cache:
            self._cache[round_id] = self.provider(round_id)

    def __call__(self, round_id: int) -> np.ndarray:
        cached = self._cache.pop(round_id, None)
        return self.provider(round_id) if cached is None else cached


def make_update_provider(args: argparse.Namespace):
    """Bank profiles for real runs (contract Section 3); synthetic ones only for smoke runs."""
    if args.value_bank:
        if args.clients is None:
            raise ValueError("--value-bank needs --clients (the bank is validated for N clients)")
        return bank_update_provider(load_bank(args.value_bank, args.clients), args.seed, args.client_id)
    profiles = profile_bank(args.seed + args.client_id)

    def synthetic_provider(round_id: int) -> np.ndarray:
        profile = profiles[(round_id - 1) % len(profiles)]
        return synthetic_update(profile, seed=args.seed + args.client_id * 1009 + round_id)

    return synthetic_provider


def load_factory(spec: str):
    """'package.module:function' -> the function."""
    import importlib

    module, sep, name = str(spec).partition(":")
    if not sep or not module or not name:
        raise ValueError(f"factory must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def make_fl_updates(args: argparse.Namespace):
    """M2: train Delta_r on the FL server's MODEL r in the background (fl_service.TrainedUpdates)."""
    from fedgrant.mininet.fl_service import TrainedUpdates, make_femnist_trainer
    from fedgrant.policy.constants import D_MODEL

    if args.fl_trainer:
        trainer = load_factory(args.fl_trainer)(client_id=args.client_id, seed=args.seed)
    else:
        trainer = make_femnist_trainer(args.client_id, seed=args.seed, data_root=args.fl_data_root,
                                       shard_root=args.fl_shard_root)
    return TrainedUpdates(trainer, d=D_MODEL, wait_s=args.fl_wait_s)


async def run(args: argparse.Namespace) -> None:
    fl_mode = getattr(args, "fl_model_port", None) is not None
    model_receiver = None
    if fl_mode:
        # M2 (end-to-end FL): the update is trained on the received global model,
        # and error feedback carries undelivered blocks to the next round (D-19).
        from fedgrant.mininet.fl_service import start_model_receiver

        update_provider = make_fl_updates(args)
        model_receiver = await start_model_receiver(update_provider, port=args.fl_model_port)
        core = ClientCore(args.client_id, error_feedback=True)
    else:
        update_provider = PrefetchedUpdates(make_update_provider(args), args.rounds)
        update_provider.prefetch(1)
        # Network-only Mininet runs (M1, smoke): each round's update is the bank
        # profile itself, so error feedback stays off (decision D-19, as in the B1-B3
        # clients). With it on, undelivered residuals would change the READY energies
        # away from the profile the M-sim twin receives.
        core = ClientCore(args.client_id, error_feedback=False)

    packet_path = Path(args.log_file).with_name(f"client_{args.client_id}_packets.csv")
    packet_stream = packet_path.open("w", newline="", encoding="utf-8")
    packet_writer = csv.DictWriter(packet_stream, fieldnames=PACKET_COLUMNS, extrasaction="raise")
    packet_writer.writeheader()

    def observe_packet(data: bytes, timestamp_ns: int) -> None:
        try:
            current_l = core.round.L if core.round is not None else None
            msg = decode_message(data, block_length=current_l)
            packet_type = MessageType(msg.message_type).name
            body = msg.body
            packet_writer.writerow({
                "t_mono_ns": timestamp_ns, "round_id": msg.round_id, "direction": "sent",
                "client_id": args.client_id, "packet_seq": "", "grant_id": body.get("grant_id", ""),
                "send_pos": body.get("send_pos", ""), "block_idx": body.get("block_idx", ""),
                "packet_type": packet_type, "bytes": len(data),
                "payload_bytes": (2 * current_l if packet_type == "DATA" and current_l is not None else 0),
                "result": "sent",
            })
            packet_stream.flush()
        except (WireProtocolError, ValueError, TypeError):
            return

    client = await UdpClient.bind(core, gateway_addr=("10.0.0.254", args.port),
                                  update_provider=update_provider, packet_observer=observe_packet)
    completed = []
    try:
        last_seen = 0
        for _ in range(args.rounds):
            try:
                round_id = await client.wait_for_round_end(timeout=args.round_timeout_s)
            except asyncio.TimeoutError:
                break
            if round_id > last_seen:
                completed.append(round_id)
                last_seen = round_id
                # Idle until the next ROUND_START (late drain + Tc): prepare its update now.
                # (M2 trains as soon as the next model arrives instead.)
                if hasattr(update_provider, "prefetch"):
                    update_provider.prefetch(round_id + 1)
            # Event is level-triggered and remains set after ROUND_END. Consume
            # it here so the next wait cannot return the previous round twice.
            client._round_end_event.clear()
        stats = client.stats
        summary = {
            "client_id": args.client_id, "completed_rounds": completed, "sent": stats.sent,
            "received": stats.received, "malformed": stats.malformed,
            "timeouts": stats.round_end_timeouts,
        }
        if fl_mode:
            summary.update(fl_models_received=update_provider.models_received,
                           fl_missing_models=update_provider.missing_models)
        Path(args.log_file).write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        try:
            await client.close()
        finally:
            packet_stream.close()
            if model_receiver is not None:
                await model_receiver.close()
            if fl_mode:
                update_provider.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-timeout-s", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--clients", type=int, default=None, help="N; required with --value-bank")
    parser.add_argument("--value-bank", default=None,
                        help="FULL value bank (.npz); omit only for synthetic smoke profiles")
    fl = parser.add_argument_group("M2 (end-to-end FL)")
    fl.add_argument("--fl-model-port", type=int, default=None,
                    help="receive the FL server's model on this TCP port and train locally (M2)")
    fl.add_argument("--fl-data-root", default="data/femnist/reproduced/leaf/data")
    fl.add_argument("--fl-shard-root", default="data/femnist/reproduced/clients")
    fl.add_argument("--fl-trainer", default=None,
                    help="module:function trainer factory (tests); default: the FEMNIST CNN trainer")
    fl.add_argument("--fl-wait-s", type=float, default=30.0,
                    help="max wait at ROUND_START for local training still running")
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
