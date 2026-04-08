FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./server /app/server
COPY ./static /app/static
COPY ./env /app/env
COPY ./openenv.yaml /app/openenv.yaml
COPY ./inference.py /app/inference.py
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
