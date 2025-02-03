FROM python:3.12

WORKDIR /app

ENV FASTAPI_ENV=production
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy the rest of the application
COPY . .

RUN poetry --version
# Copy requirements first to leverage Docker cache
RUN poetry install --no-interaction --no-ansi

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "narrativeai.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

