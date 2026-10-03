# OPTIONAL. bdopt does not need Docker: `pip install bdopt` + any EnergyPlus install is enough.
# This image just bundles both, for one EnergyPlus version per image:
#
#   docker build -f docker/Dockerfile --build-arg EPLUS_VERSION=9.6 -t bdopt:eplus-9.6 .
#   docker run --rm -v "$PWD:/work" bdopt:eplus-9.6 bdopt run study.yaml
#
# EPLUS_VERSION: any version listed by `python tools/install_eplus.py --list` (8.7 ... latest).
# Ubuntu 22.04 provides the glibc/libstdc++ needed by every release from 8.7 to 25.x.
FROM ubuntu:22.04

ARG EPLUS_VERSION=9.6
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    BDOPT_EPLUS_PATH=/opt/eplus

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates libx11-6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bdopt
COPY tools/install_eplus.py /opt/bdopt/tools/install_eplus.py
RUN python3 tools/install_eplus.py "${EPLUS_VERSION}" --dir /opt/eplus

COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install ".[torch,viz,yaml]"
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /work
CMD ["bdopt", "eplus", "list"]
