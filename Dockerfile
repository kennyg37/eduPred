FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY . .
CMD ["python", "app.py"]