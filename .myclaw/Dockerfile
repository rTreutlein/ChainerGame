FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV PIP_CACHE_DIR=/cache/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir "setuptools>=68" \
    && mkdir -p "${PIP_CACHE_DIR}" \
    && chmod a+rwx "${PIP_CACHE_DIR}"
