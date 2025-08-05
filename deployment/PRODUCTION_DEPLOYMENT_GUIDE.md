# Production Deployment Guide

## Deepfake Detection System - Production Deployment

This guide provides comprehensive instructions for deploying the Deepfake Detection System in a production environment with high availability, security, and monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Security Configuration](#security-configuration)
4. [Deployment Options](#deployment-options)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Docker Compose Deployment](#docker-compose-deployment)
7. [Monitoring Setup](#monitoring-setup)
8. [Database Configuration](#database-configuration)
9. [SSL/TLS Configuration](#ssltls-configuration)
10. [Backup and Recovery](#backup-and-recovery)
11. [Scaling and Performance](#scaling-and-performance)
12. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Kubernetes Cluster**: v1.20+ with at least 3 nodes
- **Docker**: v20.10+ for containerization
- **Helm**: v3.0+ for package management
- **kubectl**: v1.20+ for cluster management
- **Storage**: Fast SSD storage for models and uploads
- **Memory**: Minimum 8GB RAM per node
- **CPU**: Minimum 4 cores per node (8+ recommended)

### Software Dependencies

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Ingress/Nginx │    │   API Gateway   │
│   (Cloud/On-prem)│   │   (SSL/TLS)     │    │   (Rate Limiting)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Deepfake API  │    │   Grafana       │
│   (Monitoring)  │    │   (Auto-scaling)│    │   (Dashboard)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │    │   Model Storage │
│   (Primary DB)  │    │   (Sessions)    │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Security Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
DB_PASSWORD=your_secure_db_password_here
DATABASE_URL=postgresql://deepfake_user:${DB_PASSWORD}@postgres:5432/deepfake_db

# Security Keys
SECRET_KEY=your_very_long_random_secret_key_here
JWT_SECRET=your_jwt_secret_key_here

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Monitoring
GRAFANA_PASSWORD=your_grafana_admin_password

# SSL/TLS (if using self-signed certificates)
SSL_CERT_FILE=/etc/nginx/ssl/cert.pem
SSL_KEY_FILE=/etc/nginx/ssl/key.pem

# API Configuration
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE=100MB
```

### Security Best Practices

1. **Use Strong Passwords**: Generate random passwords for all services
2. **Enable SSL/TLS**: Always use HTTPS in production
3. **Network Policies**: Restrict pod-to-pod communication
4. **RBAC**: Implement proper role-based access control
5. **Secrets Management**: Use Kubernetes secrets or external secret managers
6. **Regular Updates**: Keep all components updated

## Deployment Options

### Option 1: Kubernetes Deployment (Recommended)

For production environments with high availability requirements.

### Option 2: Docker Compose Deployment

For simpler deployments or development environments.

## Kubernetes Deployment

### Step 1: Prepare the Cluster

```bash
# Create namespace
kubectl create namespace deepfake-detection

# Create monitoring namespace
kubectl create namespace monitoring
```

### Step 2: Configure Secrets

Update `deployment/k8s/secret.yaml` with your actual values:

```bash
# Generate base64 encoded secrets
echo -n "your_secret_key_here" | base64
echo -n "your_db_password_here" | base64
echo -n "your_grafana_password_here" | base64
```

### Step 3: Deploy Infrastructure

```bash
# Apply all Kubernetes resources
kubectl apply -f deployment/k8s/

# Verify deployment
kubectl get all -n deepfake-detection
```

### Step 4: Deploy Monitoring Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --set prometheus.prometheusSpec.retention=7d \
    --set grafana.adminPassword=$GRAFANA_PASSWORD
```

### Step 5: Configure Ingress

Update the ingress configuration with your domain:

```yaml
# In deployment/k8s/ingress.yaml
spec:
  rules:
  - host: api.yourdomain.com  # Replace with your domain
```

### Step 6: Deploy Application

```bash
# Build and push Docker image
docker build -t your-registry.com/deepfake-detection:latest .
docker push your-registry.com/deepfake-detection:latest

# Deploy using the deployment script
./deployment/deploy.sh k8s
```

## Docker Compose Deployment

### Step 1: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### Step 2: Deploy Services

```bash
# Deploy all services
docker-compose up -d

# Check service status
docker-compose ps
```

### Step 3: Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Check monitoring
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana
```

## Monitoring Setup

### Prometheus Configuration

The system includes comprehensive monitoring with:

- **API Metrics**: Response times, error rates, request volumes
- **System Metrics**: CPU, memory, disk usage
- **Database Metrics**: Connection pools, query performance
- **Model Metrics**: Processing times, accuracy rates

### Grafana Dashboards

Access Grafana at `http://your-domain:3000` with:
- Username: `admin`
- Password: Set in environment variables

### Alerting

Configure alerts for:
- High error rates (>5%)
- High response times (>2s)
- Low disk space (<10%)
- Service outages

## Database Configuration

### PostgreSQL Optimization

The deployment includes optimized PostgreSQL settings:

```sql
-- Connection pooling
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB

-- Performance tuning
checkpoint_completion_target = 0.9
wal_buffers = 16MB
random_page_cost = 1.1
```

### Backup Strategy

```bash
# Automated backups
kubectl create -f deployment/backup-cronjob.yaml

# Manual backup
./deployment/deploy.sh backup
```

## SSL/TLS Configuration

### Using Let's Encrypt (Recommended)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.8.0/cert-manager.yaml

# Create cluster issuer
kubectl apply -f deployment/ssl/cluster-issuer.yaml
```

### Using Self-Signed Certificates

```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout deployment/nginx/ssl/key.pem \
    -out deployment/nginx/ssl/cert.pem
```

## Backup and Recovery

### Automated Backups

```yaml
# deployment/backup-cronjob.yaml
apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U deepfake_user deepfake_db > /backup/backup-$(date +%Y%m%d).sql
```

### Recovery Procedures

```bash
# Restore from backup
kubectl exec -it postgres-pod -- psql -U deepfake_user deepfake_db < backup-file.sql

# Point-in-time recovery
kubectl exec -it postgres-pod -- pg_restore -U deepfake_user -d deepfake_db backup-file.dump
```

## Scaling and Performance

### Horizontal Pod Autoscaling

The system automatically scales based on:
- CPU usage (>70%)
- Memory usage (>80%)
- Custom metrics

### Performance Optimization

1. **Model Caching**: Models are cached in memory for faster inference
2. **Connection Pooling**: Database and Redis connections are pooled
3. **CDN Integration**: Static assets served via CDN
4. **Load Balancing**: Requests distributed across multiple pods

### Resource Limits

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod logs
kubectl logs -f deployment/deepfake-api -n deepfake-detection

# Check pod status
kubectl describe pod -l app=deepfake-api -n deepfake-detection
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it postgres-pod -- psql -U deepfake_user -d deepfake_db -c "SELECT 1;"

# Check database logs
kubectl logs -f deployment/postgres -n deepfake-detection
```

#### 3. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n deepfake-detection

# Analyze memory usage
kubectl exec -it deepfake-api-pod -- python -c "import psutil; print(psutil.virtual_memory())"
```

#### 4. Slow Response Times

```bash
# Check API metrics
curl http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))

# Check database performance
kubectl exec -it postgres-pod -- psql -U deepfake_user -d deepfake_db -c "SELECT * FROM pg_stat_activity;"
```

### Log Analysis

```bash
# View application logs
kubectl logs -f deployment/deepfake-api -n deepfake-detection

# View nginx logs
kubectl logs -f deployment/nginx -n deepfake-detection

# View monitoring logs
kubectl logs -f deployment/prometheus -n monitoring
```

### Performance Monitoring

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n deepfake-detection

# Monitor API performance
curl http://localhost:8000/metrics | grep http_request_duration
```

## Maintenance

### Regular Maintenance Tasks

1. **Database Maintenance**: Weekly vacuum and analyze
2. **Log Rotation**: Daily log cleanup
3. **Security Updates**: Monthly security patches
4. **Backup Verification**: Weekly backup testing
5. **Performance Tuning**: Monthly performance review

### Update Procedures

```bash
# Update application
./deployment/deploy.sh k8s

# Update monitoring stack
helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring

# Update database
kubectl exec -it postgres-pod -- psql -U deepfake_user -d deepfake_db -f migration.sql
```

## Support and Documentation

For additional support:

1. **Logs**: Check application and system logs
2. **Metrics**: Monitor Grafana dashboards
3. **Documentation**: Review API documentation at `/docs`
4. **Health Checks**: Use `/health` endpoint for system status

## Security Checklist

- [ ] SSL/TLS certificates configured
- [ ] Strong passwords set for all services
- [ ] Network policies implemented
- [ ] RBAC configured
- [ ] Secrets properly managed
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Regular security updates scheduled
- [ ] Backup encryption enabled
- [ ] Monitoring and alerting configured

This deployment guide ensures a production-ready, secure, and scalable deepfake detection system.