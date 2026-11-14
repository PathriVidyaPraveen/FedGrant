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
# Debian's mininet package installs into the system python3 (3.11) dist-packages,
# which the image's /usr/local python does not search. Expose only that package.
RUN ln -s /usr/lib/python3/dist-packages/mininet "$(python -c 'import site; print(site.getsitepackages()[0])')/mininet" \
    && python -c "import mininet.net"
COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e . && chmod +x docker/m1_repro.sh

# M1 reproduction (privileged): see docs/DOCKER_M1_REPRODUCTION.md.
# Full suite. Root-only Mininet tests skip here: a container shares the host
# kernel, so they need a privileged container and host sch_netem/sch_tbf.
# test_fl_smoke needs the regenerated FEMNIST data (never baked into the image;
# spec 5.1); run it with `make test-ml` after `scripts/get_femnist.sh`.
CMD ["python", "-m", "pytest", "-q", "--ignore=tests/integration/test_fl_smoke.py"]
