#!/bin/bash

# Production Deployment Script for Deepfake Detection System
# This script handles the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="deepfake-detection"
NAMESPACE="deepfake-detection"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="latest"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "docker is not installed"
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        error "helm is not installed"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log "Creating namespace $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    log "Prerequisites check completed"
}

# Build and push Docker image
build_image() {
    log "Building Docker image..."
    
    # Build the image
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG .
    
    # Push to registry
    log "Pushing image to registry..."
    docker push $DOCKER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG
    
    log "Image build and push completed"
}

# Deploy to Kubernetes
deploy_k8s() {
    log "Deploying to Kubernetes..."
    
    # Apply namespace
    kubectl apply -f deployment/k8s/namespace.yaml
    
    # Apply secrets (make sure to update with actual values)
    kubectl apply -f deployment/k8s/secret.yaml
    
    # Apply configmap
    kubectl apply -f deployment/k8s/configmap.yaml
    
    # Apply persistent volumes
    kubectl apply -f deployment/k8s/persistent-volumes.yaml
    
    # Apply deployment
    kubectl apply -f deployment/k8s/deployment.yaml
    
    # Apply service
    kubectl apply -f deployment/k8s/service.yaml
    
    # Apply HPA
    kubectl apply -f deployment/k8s/hpa.yaml
    
    # Apply ingress
    kubectl apply -f deployment/k8s/ingress.yaml
    
    log "Kubernetes deployment completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set prometheus.prometheusSpec.retention=7d \
        --set grafana.adminPassword=$GRAFANA_PASSWORD
    
    # Deploy custom Prometheus config
    kubectl apply -f deployment/monitoring/prometheus.yml -n monitoring
    
    log "Monitoring stack deployment completed"
}

# Deploy with Docker Compose (alternative)
deploy_docker_compose() {
    log "Deploying with Docker Compose..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        warn "Creating .env file from template"
        cp .env.example .env
        warn "Please update .env file with actual values before continuing"
        read -p "Press Enter to continue after updating .env file..."
    fi
    
    # Deploy services
    docker-compose up -d
    
    log "Docker Compose deployment completed"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=deepfake-api -n $NAMESPACE --timeout=300s
    
    # Check API health
    API_URL=$(kubectl get ingress deepfake-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$API_URL" ]; then
        log "Checking API health at $API_URL"
        curl -f http://$API_URL/health || error "API health check failed"
    else
        warn "Could not determine API URL, skipping health check"
    fi
    
    log "Health checks completed"
}

# Backup database
backup_database() {
    log "Creating database backup..."
    
    # Create backup directory
    mkdir -p backups/$(date +%Y%m%d_%H%M%S)
    
    # Backup PostgreSQL
    kubectl exec -n $NAMESPACE deployment/deepfake-postgres -- pg_dump -U deepfake_user deepfake_db > backups/$(date +%Y%m%d_%H%M%S)/database_backup.sql
    
    log "Database backup completed"
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    # Rollback to previous deployment
    kubectl rollout undo deployment/deepfake-api -n $NAMESPACE
    
    log "Rollback completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up resources..."
    
    # Remove old images
    docker image prune -f
    
    # Remove old backups (keep last 7 days)
    find backups -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    
    log "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting production deployment..."
    
    # Parse command line arguments
    case "${1:-}" in
        "k8s")
            check_prerequisites
            build_image
            deploy_k8s
            deploy_monitoring
            health_check
            ;;
        "docker")
            deploy_docker_compose
            health_check
            ;;
        "backup")
            backup_database
            ;;
        "rollback")
            rollback
            ;;
        "cleanup")
            cleanup
            ;;
        "full")
            check_prerequisites
            build_image
            deploy_k8s
            deploy_monitoring
            health_check
            backup_database
            cleanup
            ;;
        *)
            echo "Usage: $0 {k8s|docker|backup|rollback|cleanup|full}"
            echo "  k8s     - Deploy to Kubernetes"
            echo "  docker  - Deploy with Docker Compose"
            echo "  backup  - Create database backup"
            echo "  rollback- Rollback to previous deployment"
            echo "  cleanup - Clean up old resources"
            echo "  full    - Complete deployment with backup and cleanup"
            exit 1
            ;;
    esac
    
    log "Deployment completed successfully!"
}

# Run main function
main "$@" 