# 🚀 Fake News Detection - Deployment Guide

## 📋 Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring & Maintenance](#monitoring--maintenance)

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd fake_news_detector_end_to_end

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Training the Model
```bash
# Prepare data
python data_preparation.py

# Train model
python model_training.py

# Evaluate model
python evaluation.py

# Run complete pipeline
python run_all.py
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t fake-news-detector .

# Run container
docker run -p 7860:7860 fake-news-detector

# Or use docker-compose
docker-compose up -d
```

### Environment Variables
```bash
# Optional environment variables
TRANSFORMERS_CACHE=/app/cache
PYTHONPATH=/app
```

## ☁️ Cloud Deployment

### 1. Hugging Face Spaces
```bash
# Create a new Space on https://huggingface.co/spaces
# Upload files:
# - app.py
# - requirements.txt
# - config.py
# - utils.py
# - simple_inference.py

# Add these files to your Space:
```

**requirements.txt for Spaces:**
```
transformers==4.35.0
torch==2.1.0
gradio==4.7.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
wordcloud==1.9.2
```

### 2. Google Cloud Platform

#### App Engine Deployment
```yaml
# app.yaml
runtime: python39

env_variables:
  PYTHONPATH: /srv

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 4
```

```bash
# Deploy to App Engine
gcloud app deploy
```

#### Cloud Run Deployment
```bash
# Build and push to Container Registry
docker build -t gcr.io/YOUR_PROJECT_ID/fake-news-detector .
docker push gcr.io/YOUR_PROJECT_ID/fake-news-detector

# Deploy to Cloud Run
gcloud run deploy fake-news-detector \
  --image gcr.io/YOUR_PROJECT_ID/fake-news-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### 3. Amazon Web Services (AWS)

#### Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize application
eb init

# Create environment
eb create production

# Deploy
eb deploy
```

#### ECS Deployment
```bash
# Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service --cluster your-cluster --service-name fake-news-detector
```

### 4. Microsoft Azure

#### Container Instances
```bash
# Create resource group
az group create --name fake-news-rg --location eastus

# Deploy container
az container create \
  --resource-group fake-news-rg \
  --name fake-news-detector \
  --image your-registry/fake-news-detector \
  --ports 7860 \
  --memory 4 \
  --cpu 2
```

## 🔒 Production Considerations

### Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication if needed
- Validate all inputs
- Use environment variables for secrets

### Performance
- Use GPU instances for better performance
- Implement caching for model predictions
- Use CDN for static assets
- Monitor memory usage
- Implement horizontal scaling

### Configuration
```python
# production_config.py
import os

class ProductionConfig:
    MODEL_CACHE_SIZE = 1000
    MAX_BATCH_SIZE = 50
    RATE_LIMIT = "100/hour"
    LOG_LEVEL = "INFO"
    USE_GPU = True if torch.cuda.is_available() else False
```

### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server fake-news-detector:7860;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## 📊 Monitoring & Maintenance

### Health Checks
```python
# health_check.py
import requests
import time

def check_health():
    try:
        response = requests.get("http://localhost:7860/health", timeout=10)
        return response.status_code == 200
    except:
        return False
```

### Logging
```python
# logging_config.py
import logging
import sys

def setup_production_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/logs/app.log', rotation='midnight')
        ]
    )
```

### Metrics Collection
- Monitor API response times
- Track prediction accuracy over time
- Monitor resource usage (CPU, memory, GPU)
- Log user interaction patterns

### Backup Strategy
- Regular model backups
- Database backups (if applicable)
- Configuration backups
- Log archival

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model files
   ls -la models/
   
   # Verify dependencies
   pip list | grep transformers
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Increase container memory
   docker run -m 4g fake-news-detector
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep 7860
   
   # Use different port
   python app.py --port 8080
   ```

### Performance Optimization
- Use model quantization for smaller models
- Implement request batching
- Use async processing for multiple predictions
- Cache frequent predictions

## 📞 Support

For technical issues:
1. Check logs: `docker logs <container-id>`
2. Verify model files exist
3. Test with simple inputs first
4. Check resource limits

## 🔄 Updates & Maintenance

### Model Updates
```bash
# Backup current model
cp -r models/ models_backup/

# Update model
python model_training.py

# Test new model
python evaluation.py

# Deploy updated model
docker-compose up -d --build
```

### Application Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
docker-compose down
docker-compose up -d --build
```

This deployment guide covers local development to production deployment across major cloud platforms. Choose the deployment method that best fits your needs and infrastructure requirements.