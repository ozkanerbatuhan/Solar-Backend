FROM python:3.10-slim

WORKDIR /app

# Gerekli paketleri kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Uygulama portunu dışa aç
EXPOSE 8000

# Uygulamayı çalıştır
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 