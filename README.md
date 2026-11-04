# FedGrant
This is a GitHub repository for CS3530 Computer Networks Project. This project is about designing an Online, Receiver-Driven Allocation of Deadline Bounded Federated Learning Uplinks under Bursty Loss.

## FEMNIST data

The project specification uses LEAF commit `09ec454a5675e32e1f0546b456b77857fdece018` and these preprocessing settings:

```bash
./preprocess.sh -s niid -sf 0.1 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796
```

From the repository root, prepare a reproducible copy with:

```bash
bash scripts/get_femnist.sh
```

The script writes to `data/femnist/reproduced/` and refuses to overwrite that path. Pass another output path as its first argument to prepare a separate copy. The output contains the LEAF train/test JSON files, memory-mappable per-client NumPy shards, a client manifest, and `provenance.json` with the exact commit and settings. It selects the first 200 sorted writer IDs and groups 10 consecutive writers into each of 20 clients. Images are 28×28 grayscale values in `[0,1]`, with no augmentation. The source `data/femnist/clients/` and any existing runs are left untouched.

To run a full-participation reference experiment on that generated dataset, pass its paths explicitly, for example:

```bash
.venv/bin/python -m fedgrant.fl.run_fl --seed 0 \
  --data-root data/femnist/reproduced/leaf/data \
  --shard-root data/femnist/reproduced/clients \
  --bank data/bank/femnist_full_s0.npz
```

The FL runner defaults to those same `reproduced/` paths and refuses datasets
without embedded provenance unless `--allow-unverified-data` is explicitly
passed. Every run record stores the preprocessing version, the provenance
document and its SHA-256, the client-manifest hash, and the per-shard hashes.
Value banks store the same dataset metadata. Existing runs and banks created
from `data/femnist/clients/` remain unverified; their data source cannot be
recovered from the artifacts alone.

## Simulator ML splits

Generate the four event/feature datasets from the discrete-event simulator:

```bash
.venv/bin/python -m fedgrant.ml.datagen
```

By default this uses every seed in the split ranges from
`fedgrant.ml.schema.SPLITS`, runs 50 rounds per seed, and writes bounded shards
under `data/mldata/{train,val,test_id,test_ood}/`. In-distribution splits use
the required `*_train.csv`, `*_val.csv`, and `*_test_id.csv` cellular traces;
the OOD split uses `*_test_ood.csv` OOD traces. Missing trace files are reported
before generation rather than silently replaced by constant-rate links. Use
`--runs-per-split 1` for a small smoke run, `--trace-dir` to locate traces, or
`--out` to choose a new output directory. The generator will not overwrite
existing shards.

The simulator and gateway feature code share `fedgrant.policy.features.features48`.
The parity integration test checks that `GatewaySim` imports that exact
implementation. This checkout does not contain a separate gateway-daemon
implementation, so parity against a deployed daemon still requires running the
same test vectors against that external implementation.
