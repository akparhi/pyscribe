FROM python:latest
RUN apt-get update -y
RUN apt-get install -y ffmpeg

ENV PORT 8000

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# Epose port & run the app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
