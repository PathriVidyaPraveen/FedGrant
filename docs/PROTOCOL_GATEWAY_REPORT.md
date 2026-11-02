# Protocol and Gateway Report

## Status and scope

This report describes the implementation and test coverage present in this
checkout. Passing unit or loopback tests establish software behavior at those
boundaries; they do not establish production FL integration or Mininet
experimental results. The profiling procedure below is optional and its
measurements have not been run on a production gateway.

## Wire protocol

The wire codec defines six message types using a 20-byte network-order common
header (`!HBBIHHQ`): `ROUND_START`, `READY`, `GRANT`, `DATA`, `ROUND_END`, and
`PLAN`. The header carries magic, version, type, round ID, client ID, flags,
and sender monotonic timestamp. Message bodies and reserved-field validation
are declared in `fedgrant/common/wire.py`; encode/decode reject malformed,
truncated, trailing, or invalid-value fields.

`ROUND_START` communicates the deadline, block count and length, READY delay
(`rho_us`), and transfer mode. `READY` reports the client's ordered energy
values. In grant mode, `GRANT` identifies a grant and 1-based send position.
`DATA` carries the grant or push/PLAN sentinel, send position, original block
index, true coordinate count, and FP16 coordinates. `ROUND_END` contains a
block-index delivery bitmap. B3 uses the existing `PLAN` message for its
planned block count and rate. No additional message type or ACK mechanism is
introduced by gateway profiling.

## Gateway architecture and round lifecycle

`GatewayCore` is the socket-independent state machine. It validates round and
client identity, READY values, DATA shape/content, grant ownership, send
positions, deadlines, and duplicate delivery. Selector, eligibility, pacing,
timeout, prediction, estimator, integrity-check, and aggregation behavior enter
through provider interfaces; the core does not implement those algorithms.

At round start, the core resets per-round client state and emits a
`ROUND_START` action for each client. In grant mode, accepted READY data makes
a client eligible for scheduling. The core asks the injected policy and pacing
providers for the next eligible client, assigns a new grant ID and 1-based send
position, records the timeout, and returns a `GRANT` action. A valid DATA
packet resolves its matching grant and records the delivered original block.
Expired grants are resolved as losses on scheduler ticks; when retransmission
is enabled, timed-out positions can become eligible again. At the deadline the
core stops issuing grants, resolves outstanding grants as losses, sends each
client a `ROUND_END` delivery bitmap, and invokes the aggregation boundary if
configured. Late or invalid packets do not reopen completed round state.

`UdpGateway` owns asyncio datagram I/O, endpoint checks, action serialization,
and a stable-cadence tick task. `UdpClient` handles `ROUND_START`, delayed
READY, grant-directed DATA, and `ROUND_END`; if the latter is missing, it
finishes the round after the specified 200 ms post-deadline grace period.
Gateway and client adapters can be exercised over localhost, but the
testbed's current Mininet process composition defaults to a stub gateway and
synthetic client updates. It is not a production FL server/client integration.

## Optional gateway profiling

Pass `--profile-gateway` to the Mininet smoke runner to collect profiles;
without the flag, process sampling and scheduler-latency collection are
disabled. The process sampler attaches to the gateway child process. Each
`processes.csv` record includes monotonic timestamp, PID, cumulative user-plus-
system process CPU time in seconds, RSS in bytes, and thread count. After the
first sample, it records the monotonic wall interval and
`100 * process CPU-time delta / wall-time delta`, expressed as percent of one
logical CPU (so a multithreaded process can exceed 100%).

Scheduler samples measure monotonic wall time from either UDP datagram callback
entry or scheduler-tick entry until a generated GRANT action reaches the
gateway action-dispatch boundary. `scheduler_samples.csv` includes the trigger
type and round ID. These are software scheduling latencies, not packet RTT or
wire-transmission latency; the metric does not include time spent in the
network after the action is dispatched.

`gateway_profile.json` excludes the first 20 rounds. Process CPU/RSS samples
are filtered at the beginning of round 21, and CPU intervals that overlap the
warm-up boundary are discarded. Scheduler rows are filtered by round ID.
Summaries report p50 when at least one sample exists, p95 at 20 or more
samples, and p99 at 100 or more samples; unsupported percentiles and empty
sample summaries are `null`, not zero. RSS is reported in bytes and MiB.
`meta.json` records whether profiling was enabled, the gateway process role,
warm-up count, and the run's scenario, seed, client count, beta, gateway
factory, and timing configuration.

### Reproduction command

On a host with the project's Mininet/OVS prerequisites and required privileges,
the following collects profiling data for the existing **stub smoke path**:

```sh
sudo .venv/bin/python -m fedgrant.mininet.run_mininet \
  --clients 3 --rounds 21 --scenario L2 --seed 0 --beta 0.10 \
  --profile-gateway --output results/mininet/task18-profile-smoke
```

The output directory contains `processes.csv`, `scheduler_samples.csv`,
`gateway_profile.json`, `gateway_summary.json`, and `meta.json`, alongside the
existing packet, queue, and process logs. Twenty warm-up rounds leave one
post-warm-up round in this example. For stable tail percentiles, use enough
post-warm-up rounds and samples to meet the thresholds above. The stub run is
an instrumentation smoke test only: it is not M2, does not profile production
policy dependencies, and must not be reported as production FL performance.

## Validation and limitations

Focused deterministic tests cover CPU-time/wall-time conversion, process CSV
fields and units, warm-up filtering, percentile availability, empty samples,
serialization, and measured datagram/tick-to-GRANT latency. The focused
profiling and Mininet tests passed (**12 passed**); the relevant protocol,
gateway, baseline, Task 16, Task 17, and Mininet regression set passed
(**333 passed, 1 skipped**). The skip is the host-gated Mininet topology test.
`compileall` and `git diff --check` also passed. These checks do not replace
production integration or experimental validation.

No M2 experiment or production gateway profile has been run from this
environment. The available Mininet runner lacks production FL lifecycle
wiring, and execution requires host networking/privilege prerequisites. No
profiling measurements or M2 results are claimed here.
