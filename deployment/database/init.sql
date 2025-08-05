-- PostgreSQL initialization script for Deepfake Detection System
-- Production-optimized configuration

-- Create database if not exists
SELECT 'CREATE DATABASE deepfake_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'deepfake_db')\gexec

-- Connect to the database
\c deepfake_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    api_key VARCHAR(255) UNIQUE,
    rate_limit_per_minute INTEGER DEFAULT 100
);

-- Create uploads table
CREATE TABLE IF NOT EXISTS uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    upload_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Create detection_results table
CREATE TABLE IF NOT EXISTS detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    upload_id UUID REFERENCES uploads(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    is_deepfake BOOLEAN NOT NULL,
    confidence FLOAT NOT NULL,
    processing_time FLOAT,
    model_version VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create ensemble_results table
CREATE TABLE IF NOT EXISTS ensemble_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    upload_id UUID REFERENCES uploads(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    fusion_method VARCHAR(50) NOT NULL,
    is_deepfake BOOLEAN NOT NULL,
    confidence FLOAT NOT NULL,
    uncertainty FLOAT,
    agreement_score FLOAT,
    individual_predictions JSONB,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create analysis_sessions table
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'running',
    progress FLOAT DEFAULT 0.0,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    results_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    dataset_info JSONB
);

-- Create API_usage_logs table
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time FLOAT,
    ip_address INET,
    user_agent TEXT,
    request_size BIGINT,
    response_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);

CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(upload_status);
CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_file_type ON uploads(file_type);

CREATE INDEX IF NOT EXISTS idx_detection_results_upload_id ON detection_results(upload_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_model_name ON detection_results(model_name);
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON detection_results(created_at);

CREATE INDEX IF NOT EXISTS idx_ensemble_results_upload_id ON ensemble_results(upload_id);
CREATE INDEX IF NOT EXISTS idx_ensemble_results_user_id ON ensemble_results(user_id);
CREATE INDEX IF NOT EXISTS idx_ensemble_results_fusion_method ON ensemble_results(fusion_method);

CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON analysis_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON analysis_sessions(status);

CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_evaluation_date ON model_performance(evaluation_date);

CREATE INDEX IF NOT EXISTS idx_api_usage_logs_user_id ON api_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_endpoint ON api_usage_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_created_at ON api_usage_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_status_code ON api_usage_logs(status_code);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_uploads_metadata_gin ON uploads USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_detection_results_metadata_gin ON detection_results USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_ensemble_results_individual_predictions_gin ON ensemble_results USING GIN (individual_predictions);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for API usage logging
CREATE OR REPLACE FUNCTION log_api_usage(
    p_user_id UUID,
    p_endpoint VARCHAR(100),
    p_method VARCHAR(10),
    p_status_code INTEGER,
    p_response_time FLOAT,
    p_ip_address INET,
    p_user_agent TEXT,
    p_request_size BIGINT,
    p_response_size BIGINT
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO api_usage_logs (
        user_id, endpoint, method, status_code, response_time,
        ip_address, user_agent, request_size, response_size
    ) VALUES (
        p_user_id, p_endpoint, p_method, p_status_code, p_response_time,
        p_ip_address, p_user_agent, p_request_size, p_response_size
    );
END;
$$ LANGUAGE plpgsql;

-- Create function for rate limiting
CREATE OR REPLACE FUNCTION check_rate_limit(
    p_user_id UUID,
    p_limit_per_minute INTEGER DEFAULT 100
)
RETURNS BOOLEAN AS $$
DECLARE
    request_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO request_count
    FROM api_usage_logs
    WHERE user_id = p_user_id
    AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 minute';
    
    RETURN request_count < p_limit_per_minute;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT CONNECT ON DATABASE deepfake_db TO deepfake_user;
GRANT USAGE ON SCHEMA public TO deepfake_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO deepfake_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO deepfake_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO deepfake_user;

-- Set up connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';

-- Reload configuration
SELECT pg_reload_conf(); 