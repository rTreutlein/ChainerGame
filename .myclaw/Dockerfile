FROM swipl:9.3.33@sha256:f744a102ee3ded4493c993b619ec76895873ff2b1a0ec196a525cb6cb0fdaf8c

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.8.22
ARG PETTACHAINER_REV=d41c7224ea80695f90c4ca1ffecc5d3a188f3c61

RUN sed -i \
        -e 's|http://deb.debian.org/debian-security|http://snapshot.debian.org/archive/debian-security/20251103T204410Z|g' \
        -e 's|http://deb.debian.org/debian|http://snapshot.debian.org/archive/debian/20251103T204410Z|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends --no-upgrade \
        build-essential \
        ca-certificates \
        git \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp/myclaw-home \
    PIP_CACHE_DIR=/cache/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_CACHE_DIR=/tmp/myclaw-uv-cache \
    PATH=/opt/pettachainer/.venv/bin:$PATH

RUN mkdir -p "$HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
    && chmod 1777 "$HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
    && python3 -m pip install --break-system-packages --no-cache-dir "uv==$UV_VERSION" \
    && git clone --filter=blob:none https://github.com/rTreutlein/PeTTaChainer.git /opt/pettachainer \
    && git -C /opt/pettachainer checkout --detach "$PETTACHAINER_REV" \
    && test "$(git -C /opt/pettachainer rev-parse HEAD)" = "$PETTACHAINER_REV" \
    && uv sync --project /opt/pettachainer --frozen --python /usr/bin/python3 \
    && uv pip install --python /opt/pettachainer/.venv/bin/python "setuptools==80.9.0" \
    && python --version \
    && swipl --version \
    && python -c 'import janus_swi; from pettachainer import PeTTaChainer; PeTTaChainer()' \
    && chmod -R a+rwX /opt/pettachainer/.venv

ENTRYPOINT []
CMD ["bash"]
