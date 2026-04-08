FROM python:3.11-slim

EXPOSE 7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    sleep 10 && \
    streamlit run app/frontend.py \
        --server.port 7860 \
        --server.address 0.0.0.0 \
        --server.headless true