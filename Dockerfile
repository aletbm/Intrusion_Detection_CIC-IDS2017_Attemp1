FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y gcc && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY src ./api
COPY models ./models

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "api"]

#docker build -t intrusion-api .
#gcloud builds submit --tag gcr.io/plucky-haven-463121-j1/intrusion-api
#gcloud run deploy intrusion-api --image gcr.io/plucky-haven-463121-j1/intrusion-api --platform managed --region us-east1 --allow-unauthenticated --port 8080
