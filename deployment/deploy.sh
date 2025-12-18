#!/bin/bash
# HealthSync AI - Production Deployment Script

set -e

echo "🚀 Starting HealthSync AI Production Deployment..."

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
BACKUP_DIR="/backup/healthsync-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/var/log/healthsync-deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker service."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if environment file exists
    if [ ! -f ".env.production" ]; then
        error "Production environment file (.env.production) not found."
    fi
    
    log "Prerequisites check completed ✅"
}

# Create backup
create_backup() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup database if running
    if docker-compose -f docker-compose.prod.yml ps | grep -q "healthsync-postgres"; then
        log "Backing up database..."
        docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U healthsync healthsync_prod > "$BACKUP_DIR/database.sql"
    fi
    
    # Backup uploaded files
    if [ -d "./backend/uploads" ]; then
        log "Backing up uploaded files..."
        cp -r ./backend/uploads "$BACKUP_DIR/"
    fi
    
    # Backup configuration
    cp .env.production "$BACKUP_DIR/"
    cp docker-compose.prod.yml "$BACKUP_DIR/"
    
    log "Backup created at $BACKUP_DIR ✅"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    docker-compose -f docker-compose.prod.yml pull
    log "Images pulled successfully ✅"
}

# Build application images
build_images() {
    log "Building application images..."
    docker-compose -f docker-compose.prod.yml build --no-cache
    log "Images built successfully ✅"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Start database service if not running
    docker-compose -f docker-compose.prod.yml up -d postgres redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 30
    
    # Run migrations
    docker-compose -f docker-compose.prod.yml run --rm backend python -m alembic upgrade head
    
    log "Database migrations completed ✅"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Stop existing services
    docker-compose -f docker-compose.prod.yml down
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    log "Services deployed ✅"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    sleep 60
    
    # Check backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "Backend health check passed ✅"
    else
        error "Backend health check failed ❌"
    fi
    
    # Check frontend health
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log "Frontend health check passed ✅"
    else
        error "Frontend health check failed ❌"
    fi
    
    # Check database connection
    if docker-compose -f docker-compose.prod.yml exec -T backend python -c "from services.db_service import test_connection; test_connection()" > /dev/null 2>&1; then
        log "Database connection check passed ✅"
    else
        error "Database connection check failed ❌"
    fi
    
    log "All health checks passed ✅"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Start monitoring services
    docker-compose -f docker-compose.prod.yml up -d prometheus grafana
    
    # Wait for services to start
    sleep 30
    
    # Import Grafana dashboards
    if [ -d "./monitoring/grafana/dashboards" ]; then
        log "Importing Grafana dashboards..."
        # Dashboard import logic would go here
    fi
    
    log "Monitoring setup completed ✅"
}

# Setup SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."
    
    # Create SSL directory if it doesn't exist
    mkdir -p ./nginx/ssl
    
    # Check if certificates exist
    if [ ! -f "./nginx/ssl/cert.pem" ] || [ ! -f "./nginx/ssl/key.pem" ]; then
        warning "SSL certificates not found. Generating self-signed certificates for development..."
        
        # Generate self-signed certificate (for development only)
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ./nginx/ssl/key.pem \
            -out ./nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        warning "Self-signed certificates generated. Replace with proper certificates for production!"
    fi
    
    log "SSL setup completed ✅"
}

# Cleanup old images and containers
cleanup() {
    log "Cleaning up old images and containers..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused volumes (be careful with this in production)
    # docker volume prune -f
    
    log "Cleanup completed ✅"
}

# Main deployment function
main() {
    log "🚀 Starting HealthSync AI deployment to $ENVIRONMENT environment"
    
    # Load environment variables
    if [ -f ".env.production" ]; then
        export $(cat .env.production | grep -v '^#' | xargs)
    fi
    
    # Run deployment steps
    check_prerequisites
    create_backup
    setup_ssl
    pull_images
    build_images
    run_migrations
    deploy_services
    setup_monitoring
    health_check
    cleanup
    
    log "🎉 HealthSync AI deployment completed successfully!"
    log "📊 Access the application at: https://localhost"
    log "📈 Access monitoring at: http://localhost:3001 (Grafana)"
    log "🔍 Access metrics at: http://localhost:9090 (Prometheus)"
    
    # Display service status
    echo ""
    log "Service Status:"
    docker-compose -f docker-compose.prod.yml ps
}

# Handle script interruption
trap 'error "Deployment interrupted by user"' INT TERM

# Run main function
main "$@"