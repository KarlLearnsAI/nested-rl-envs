FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir gradio huggingface-hub requests pydantic matplotlib python-dotenv pyyaml supabase

EXPOSE 7860

CMD ["python", "app.py"]
