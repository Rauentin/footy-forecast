FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/params.json models/params.json
COPY data/epl_2026_27_fixtures.csv data/epl_2026_27_fixtures.csv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
