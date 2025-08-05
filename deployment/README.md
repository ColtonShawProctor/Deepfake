# Production Deployment Configuration

## Deepfake Detection System - Production Ready Deployment

This directory contains all the necessary configuration files and scripts for deploying the Deepfake Detection System in a production environment with enterprise-grade security, scalability, and monitoring.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Load      │    │   Ingress   │    │   API       │        │
│  │ Balancer    │───▶│ Controller  │───▶│ Gateway     │        │
│  │ (Cloud)     │    │ (Nginx)     │    │ (FastAPI)   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Prometheus  │    │ Deepfake    │    │ Grafana     │        │
│  │ (Metrics)   │◀──▶│ API         │◀──▶│ (Dashboard) │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ PostgreSQL  │    │ Redis       │    │ Model       │        │
│  │ (Database)  │    │ (Cache)     │    │ Storage     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
deployment/
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml            # Namespace configuration
│   ├── configmap.yaml            # Application configuration
│   ├── secret.yaml               # Sensitive data
│   ├── deployment.yaml           # Application deployment
│   ├── service.yaml              # Service configuration
│   ├── ingress.yaml              # Ingress rules
│   ├── hpa.yaml                  # Horizontal Pod Autoscaler
│   └── persistent-volumes.yaml   # Storage configuration
├── nginx/                        # Nginx configuration
│   └── nginx.conf                # Production nginx config
├── monitoring/                   # Monitoring stack
│   ├── prometheus.yml            # Prometheus configuration
│   ├── alert_rules.yml           # Alert rules
│   └── grafana/                  # Grafana dashboards
│       ├── datasources/
│       └── dashboards/
├── database/                     # Database configuration
│   └── init.sql                  # Database initialization
├── production_config.py          # Production settings
├── security_middleware.py        # Security middleware
├── deploy.sh                     # Deployment script
├── docker-compose.yml            # Docker Compose config
├── Dockerfile                    # Production Dockerfile
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** (v1.20+)
2. **Docker** (v20.10+)
3. **kubectl** (v1.20+)
4. **Helm** (v3.0+)

### Option 1: Kubernetes Deployment (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd deepfake-detection

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your production values

# 3. Deploy to Kubernetes
./deployment/deploy.sh k8s
```

### Option 2: Docker Compose Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your values

# 2. Deploy with Docker Compose
./deployment/deploy.sh docker
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
DB_PASSWORD=your_secure_password
DATABASE_URL=postgresql://deepfake_user:${DB_PASSWORD}@postgres:5432/deepfake_db

# Security
SECRET_KEY=your_very_long_random_secret_key
JWT_SECRET=your_jwt_secret_key

# Redis
REDIS_URL=redis://redis:6379/0

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# API Configuration
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE=100MB
```

### Security Configuration

The deployment includes comprehensive security features:

- **Rate Limiting**: Configurable per-minute and per-hour limits
- **Authentication**: JWT tokens and API keys
- **SSL/TLS**: Automatic certificate management
- **Security Headers**: XSS protection, CSRF protection
- **Input Validation**: File type and size validation
- **Network Policies**: Pod-to-pod communication restrictions

### Performance Optimization

- **Auto-scaling**: Horizontal Pod Autoscaler (HPA)
- **Load Balancing**: Nginx with least connections
- **Caching**: Redis for session and model caching
- **Connection Pooling**: Database connection optimization
- **Resource Limits**: CPU and memory constraints

## 📊 Monitoring and Observability

### Prometheus Metrics

The system exposes comprehensive metrics:

- **API Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk usage
- **Database Metrics**: Connection pools, query performance
- **Model Metrics**: Inference time, accuracy rates

### Grafana Dashboards

Pre-configured dashboards include:

- **System Overview**: Overall system health
- **API Performance**: Request/response metrics
- **Resource Usage**: CPU, memory, disk utilization
- **Database Performance**: Query times, connections
- **Model Performance**: Inference metrics

### Alerting

Configured alerts for:

- High error rates (>5%)
- High response times (>2s)
- Low disk space (<10%)
- Service outages
- High memory usage (>85%)

## 🔒 Security Features

### Authentication & Authorization

- **JWT Tokens**: Secure token-based authentication
- **API Keys**: For programmatic access
- **Rate Limiting**: Per-client request limits
- **Session Management**: Secure session handling

### Network Security

- **SSL/TLS**: End-to-end encryption
- **Network Policies**: Pod communication restrictions
- **Ingress Security**: Request validation and filtering
- **CORS Configuration**: Cross-origin request control

### Data Protection

- **Secrets Management**: Kubernetes secrets
- **Encryption at Rest**: Database encryption
- **Input Sanitization**: File upload validation
- **Audit Logging**: Comprehensive access logs

## 📈 Scaling and Performance

### Horizontal Scaling

- **Auto-scaling**: Based on CPU and memory usage
- **Load Distribution**: Multiple API instances
- **Database Scaling**: Read replicas and connection pooling
- **Cache Scaling**: Redis cluster support

### Performance Optimization

- **Model Caching**: In-memory model storage
- **Batch Processing**: Efficient batch inference
- **Async Processing**: Background task handling
- **CDN Integration**: Static asset delivery

### Resource Management

```yaml
# Resource limits per pod
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🗄️ Database Configuration

### PostgreSQL Optimization

- **Connection Pooling**: PgBouncer configuration
- **Indexing**: Optimized database indexes
- **Partitioning**: Large table partitioning
- **Backup Strategy**: Automated daily backups

### Data Retention

- **Logs**: 30-day retention
- **Metrics**: 7-day retention
- **Backups**: 30-day retention
- **Uploads**: Configurable retention

## 🔄 Backup and Recovery

### Automated Backups

```bash
# Create backup
./deployment/deploy.sh backup

# Restore from backup
kubectl exec -it postgres-pod -- psql -U deepfake_user -d deepfake_db < backup.sql
```

### Disaster Recovery

- **Database Recovery**: Point-in-time recovery
- **Application Recovery**: Rolling updates and rollbacks
- **Configuration Recovery**: GitOps-based configuration
- **Monitoring Recovery**: Metrics and logs preservation

## 🛠️ Maintenance

### Regular Tasks

1. **Security Updates**: Monthly security patches
2. **Performance Monitoring**: Weekly performance reviews
3. **Backup Verification**: Weekly backup testing
4. **Log Rotation**: Daily log cleanup
5. **Resource Optimization**: Monthly resource review

### Update Procedures

```bash
# Update application
./deployment/deploy.sh k8s

# Update monitoring
helm upgrade prometheus prometheus-community/kube-prometheus-stack

# Rollback if needed
./deployment/deploy.sh rollback
```

## 🐛 Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl logs -f deployment/deepfake-api -n deepfake-detection
   kubectl describe pod -l app=deepfake-api -n deepfake-detection
   ```

2. **Database Connection Issues**
   ```bash
   kubectl exec -it postgres-pod -- psql -U deepfake_user -d deepfake_db -c "SELECT 1;"
   ```

3. **High Memory Usage**
   ```bash
   kubectl top pods -n deepfake-detection
   kubectl exec -it deepfake-api-pod -- python -c "import psutil; print(psutil.virtual_memory())"
   ```

### Health Checks

```bash
# API health
curl http://your-domain/health

# Database health
kubectl exec -it postgres-pod -- pg_isready -U deepfake_user

# Redis health
kubectl exec -it redis-pod -- redis-cli ping
```

## 📚 Additional Resources

- [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
- [API Documentation](../app/main.py)
- [Monitoring Setup](monitoring/)
- [Security Configuration](security_middleware.py)

## 🤝 Support

For deployment issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [production deployment guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
3. Check application logs and metrics
4. Contact the development team

## 📄 License

This deployment configuration is part of the Deepfake Detection System and follows the same license terms as the main project.

---

**Note**: This deployment configuration is designed for production use. Always test in a staging environment before deploying to production. 