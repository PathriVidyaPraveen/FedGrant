FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git iproute2 iputils-ping iperf3 ethtool tcpdump mininet openvswitch-switch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FedGrant
COPY docker/requirements.lock docker/requirements-ml.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock -r requirements-ml.lock
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

# Full suite. Root-only Mininet tests skip here: a container shares the host
# kernel, so they need a privileged container and host sch_netem/sch_tbf.
# test_fl_smoke needs the regenerated FEMNIST data (never baked into the image;
# spec 5.1); run it with `make test-ml` after `scripts/get_femnist.sh`.
CMD ["python", "-m", "pytest", "-q", "--ignore=tests/integration/test_fl_smoke.py"]
