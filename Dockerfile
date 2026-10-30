FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p /app/data /app/outputs/paper-agent

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "paper_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
