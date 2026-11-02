"""UDP client process entry point used inside a Mininet client host."""

from __future__ import annotations

import argparse
import csv
import asyncio
import json
from pathlib import Path

import numpy as np

from fedgrant.core.client_core import ClientCore
from fedgrant.mininet.profile_update import profile_bank, synthetic_update
from fedgrant.common.constants import MessageType
from fedgrant.common.logschema import PACKET_COLUMNS
from fedgrant.common.wire import decode_message, WireProtocolError
from fedgrant.net.udp_client import UdpClient


async def run(args: argparse.Namespace) -> None:
    profiles = profile_bank(args.seed + args.client_id)
    core = ClientCore(args.client_id)

    def update_provider(round_id: int) -> np.ndarray:
        profile = profiles[(round_id - 1) % len(profiles)]
        return synthetic_update(profile, seed=args.seed + args.client_id * 1009 + round_id)

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
            # Event is level-triggered and remains set after ROUND_END. Consume
            # it here so the next wait cannot return the previous round twice.
            client._round_end_event.clear()
        stats = client.stats
        Path(args.log_file).write_text(json.dumps({
            "client_id": args.client_id, "completed_rounds": completed, "sent": stats.sent,
            "received": stats.received, "malformed": stats.malformed,
            "timeouts": stats.round_end_timeouts,
        }, indent=2) + "\n")
    finally:
        try:
            await client.close()
        finally:
            packet_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-timeout-s", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
