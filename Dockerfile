# syntax=docker/dockerfile:1.7


FROM python:3.12-slim AS deps
WORKDIR /app


COPY requirements-serve.txt /app/requirements-serve.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip wheel setuptools && \
    pip install -r requirements-serve.txt

FROM python:3.12-slim AS secrets
WORKDIR /work

COPY ansible /work/ansible

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install "ansible-core>=2.16,<2.18"

ENV ANSIBLE_CONFIG=/work/ansible/ansible.cfg

RUN --mount=type=secret,id=vault_pass \
    ansible-playbook -i /work/ansible/inventory.ini /work/ansible/site.yml \
      --vault-password-file /run/secrets/vault_pass \
 && echo "--- AFTER PLAYBOOK ---" \
 && ls -la /work || true \
 && ls -la /work/out || true \
 && ls -la /work/out/secrets || true \
 && test -f /work/out/.env \
 && test -f /work/out/secrets/gsa-dvc.json


FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=deps /usr/local /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY app/ app/
COPY web/ web/
COPY config.ini .
COPY model/ model/

COPY --from=secrets /work/out/.env /app/.env
COPY --from=secrets /work/out/secrets /app/secrets

RUN test -f /app/.env || (echo ".env not found after vault render" && exit 1) && \
    test -f /app/secrets/gsa-dvc.json || (echo "secrets/gsa-dvc.json not found after vault render" && exit 1)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:4000/ || exit 1

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app.main:app"]

