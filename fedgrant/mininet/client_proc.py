"""UDP client process entry point used inside a Mininet client host."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from fedgrant.core.client_core import ClientCore
from fedgrant.mininet.profile_update import profile_bank, synthetic_update
from fedgrant.net.udp_client import UdpClient


async def run(args: argparse.Namespace) -> None:
    profiles = profile_bank(args.seed + args.client_id)
    core = ClientCore(args.client_id)

    def update_provider(round_id: int) -> np.ndarray:
        profile = profiles[(round_id - 1) % len(profiles)]
        return synthetic_update(profile, seed=args.seed + args.client_id * 1009 + round_id)

    client = await UdpClient.bind(core, gateway_addr=("10.0.0.254", args.port), update_provider=update_provider)
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
        await client.close()


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
