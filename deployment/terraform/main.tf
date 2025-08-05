# Production Infrastructure for Deepfake Detection System
# Terraform configuration for multi-region deployment with auto-scaling

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "deepfake-detection-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "secondary_regions" {
  description = "Secondary regions for multi-region deployment"
  type        = list(string)
  default     = ["us-west-2", "eu-west-1"]
}

variable "gpu_instance_type" {
  description = "GPU instance type for model serving"
  type        = string
  default     = "g4dn.xlarge"  # Tesla T4 GPU
}

variable "enable_spot_instances" {
  description = "Use spot instances for cost optimization"
  type        = bool
  default     = true
}

# Primary region provider
provider "aws" {
  region = var.primary_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "deepfake-detection"
      ManagedBy   = "terraform"
      CostCenter  = "ml-inference"
    }
  }
}

# VPC Module for networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "deepfake-detection-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  enable_dns_support = true
  
  # Enable VPC flow logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# EKS Cluster for container orchestration
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "19.15.3"
  
  cluster_name    = "deepfake-detection-cluster"
  cluster_version = "1.28"
  
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets
  
  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Node groups
  eks_managed_node_groups = {
    # CPU node group for API servers
    api_servers = {
      desired_size = 3
      min_size     = 2
      max_size     = 10
      
      instance_types = ["t3.large"]
      capacity_type  = "SPOT"
      
      labels = {
        role = "api-server"
      }
      
      taints = []
      
      update_config = {
        max_unavailable_percentage = 50
      }
    }
    
    # GPU node group for model serving
    gpu_nodes = {
      desired_size = 2
      min_size     = 1
      max_size     = 10
      
      instance_types = [var.gpu_instance_type]
      capacity_type  = var.enable_spot_instances ? "SPOT" : "ON_DEMAND"
      
      labels = {
        role = "model-server"
        "nvidia.com/gpu" = "true"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      
      # Install NVIDIA GPU drivers
      pre_bootstrap_user_data = <<-EOT
        #!/bin/bash
        curl -fsSL -o nvidia-driver.run https://us.download.nvidia.com/tesla/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run
        sh nvidia-driver.run --silent
        
        # Install NVIDIA container toolkit
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
      EOT
      
      update_config = {
        max_unavailable = 1
      }
    }
  }
  
  # Enable cluster autoscaler
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  node_security_group_additional_rules = {
    ingress_cluster_all = {
      description = "Allow all traffic from cluster"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      source_cluster_security_group = true
    }
  }
}

# RDS Aurora PostgreSQL for application database
module "rds_aurora" {
  source = "terraform-aws-modules/rds-aurora/aws"
  version = "8.3.1"
  
  name            = "deepfake-detection-db"
  engine          = "aurora-postgresql"
  engine_version  = "15.4"
  master_username = "deepfake_admin"
  
  vpc_id               = module.vpc.vpc_id
  subnets              = module.vpc.private_subnets
  create_security_group = true
  allowed_cidr_blocks  = module.vpc.private_subnets_cidr_blocks
  
  instance_class = "db.r6g.large"
  instances = {
    one = {
      instance_class = "db.r6g.large"
    }
    two = {
      instance_class = "db.r6g.large"
    }
  }
  
  # Read replicas in secondary regions
  enable_global_write_forwarding = true
  
  # Performance and optimization
  performance_insights_enabled = true
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Backup configuration
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  
  # High availability
  availability_zones = data.aws_availability_zones.available.names
  
  db_parameter_group_name         = aws_db_parameter_group.postgresql.id
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.postgresql.id
}

# Database parameter groups
resource "aws_db_parameter_group" "postgresql" {
  name   = "deepfake-aurora-db-postgres15-parameter-group"
  family = "aurora-postgresql15"
  
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
  
  parameter {
    name  = "max_connections"
    value = "1000"
  }
}

resource "aws_rds_cluster_parameter_group" "postgresql" {
  name   = "deepfake-aurora-postgres15-cluster-parameter-group"
  family = "aurora-postgresql15"
  
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
}

# ElastiCache Redis for caching
module "elasticache_redis" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "1.0.0"
  
  cluster_id               = "deepfake-cache"
  engine_version           = "7.0"
  node_type               = "cache.r6g.large"
  number_cache_clusters    = 3
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Redis configuration
  parameter = [
    {
      name  = "maxmemory-policy"
      value = "allkeys-lru"
    }
  ]
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled        = true
  
  # Automatic failover
  automatic_failover_enabled = true
  multi_az_enabled          = true
}

# S3 buckets for model storage and static files
resource "aws_s3_bucket" "model_repository" {
  bucket = "deepfake-detection-models-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_repository" {
  bucket = aws_s3_bucket.model_repository.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_repository" {
  bucket = aws_s3_bucket.model_repository.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# CloudFront CDN for static content
resource "aws_cloudfront_distribution" "cdn" {
  enabled             = true
  is_ipv6_enabled    = true
  default_root_object = "index.html"
  
  origin {
    domain_name = aws_s3_bucket.static_content.bucket_regional_domain_name
    origin_id   = "S3-static-content"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.static.cloudfront_access_identity_path
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "S3-static-content"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
  }
  
  price_class = "PriceClass_200"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# WAF for API protection
resource "aws_wafv2_web_acl" "api_protection" {
  name  = "deepfake-api-protection"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }
  
  # SQL injection protection
  rule {
    name     = "SQLiProtection"
    priority = 2
    
    action {
      block {}
    }
    
    statement {
      sqli_match_statement {
        field_to_match {
          body {}
        }
        text_transformation {
          priority = 0
          type     = "URL_DECODE"
        }
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "SQLiProtection"
      sampled_requests_enabled   = true
    }
  }
}

# Auto Scaling policies
resource "aws_autoscaling_policy" "gpu_scale_up" {
  name                   = "gpu-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = module.eks.eks_managed_node_groups.gpu_nodes.asg_name
}

resource "aws_autoscaling_policy" "gpu_scale_down" {
  name                   = "gpu-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = module.eks.eks_managed_node_groups.gpu_nodes.asg_name
}

# CloudWatch alarms for auto-scaling
resource "aws_cloudwatch_metric_alarm" "gpu_high_utilization" {
  alarm_name          = "gpu-high-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "GPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "GPU utilization is too high"
  alarm_actions       = [aws_autoscaling_policy.gpu_scale_up.arn]
  
  dimensions = {
    AutoScalingGroupName = module.eks.eks_managed_node_groups.gpu_nodes.asg_name
  }
}

# Monitoring and alerting with CloudWatch
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "deepfake-detection-production"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/EKS", "cluster_node_count", "ClusterName", module.eks.cluster_name],
            [".", "cluster_failed_node_count", ".", "."]
          ]
          period = 300
          stat   = "Average"
          region = var.primary_region
          title  = "EKS Cluster Nodes"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/EC2", "GPUUtilization", "AutoScalingGroupName", module.eks.eks_managed_node_groups.gpu_nodes.asg_name]
          ]
          period = 300
          stat   = "Average"
          region = var.primary_region
          title  = "GPU Utilization"
        }
      }
    ]
  })
}

# Outputs
output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "database_endpoint" {
  description = "RDS Aurora endpoint"
  value       = module.rds_aurora.cluster_endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache_redis.primary_endpoint_address
}

output "cdn_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.cdn.domain_name
}