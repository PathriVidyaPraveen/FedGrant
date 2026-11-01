FROM python:3.14-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git iproute2 iputils-ping iperf3 ethtool tcpdump mininet openvswitch-switch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FedGrant
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

# A container shares the host kernel: running Mininet tests requires a
# privileged container and a host kernel with sch_netem and sch_tbf available.
CMD ["python", "-m", "pytest", "-q", "tests/unit", "--ignore=tests/unit/test_simulator_fixes.py"]
