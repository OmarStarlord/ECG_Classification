FROM python:3.11-slim
WORKDIR /app
RUN pip install tensorflow==2.15.0 numpy==1.26.4 flask werkzeug pandas
COPY . /app
CMD ["python", "main.py"]