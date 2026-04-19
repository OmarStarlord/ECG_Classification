# NN Service for ECG Classification

This Spring Boot application provides a REST API gateway for ECG classification services.

## Getting Started

### Prerequisites
- Java 11
- Maven 3.6+

### Building
```bash
mvn clean package
```

### Running
```bash
java -jar target/nn-0.0.1-SNAPSHOT.jar
```

## API Endpoints

### POST /config
Configure the classification algorithm to use.
- **Parameter**: `algo` (query parameter) - Algorithm name (mlp, cnn, rnn)
- **Response**: Configuration status from IA service

### POST /classify  
Classify ECG data from uploaded file.
- **Parameter**: `datafile` (multipart) - TSV file containing ECG data
- **Response**: Classification results from IA service

## Docker

### Build
```bash
docker build -t nn .
```

### Run
```bash
docker run -p 8080:8080 nn
```

## Architecture

This service acts as a gateway between external clients and the IA (Intelligence Artificielle) service. It:
- Receives HTTP requests from clients
- Forwards requests to the IA service on port 80
- Returns responses from IA service to clients

The IA service handles the actual machine learning model operations.
