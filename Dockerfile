FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system g2mrf && adduser --system --ingroup g2mrf g2mrf

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir .

USER g2mrf

ENTRYPOINT ["g2mrf"]
CMD ["plan"]
