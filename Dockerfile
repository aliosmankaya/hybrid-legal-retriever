
FROM python:3.12-slim
 
# Set working directory
WORKDIR /app
 
# Install Poetry
RUN pip install --no-cache-dir poetry
 
# Copy dependency files first (for better layer caching)
COPY pyproject.toml poetry.lock ./
 
# Install dependencies (no virtualenv inside container, no dev deps)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev
 
# Copy the rest of the project
COPY . .