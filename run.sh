#!/bin/bash

# YOLO Stream Detection Service Runner
# This script provides convenient commands to run and manage the service

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

install_requirements() {
    log_info "Installing Python requirements..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        log_success "Requirements installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi
}

check_model() {
    log_info "Checking for YOLO model..."
    
    if [ -f "yolo11n.pt" ]; then
        log_success "YOLO model found"
    else
        log_warning "YOLO model (yolo11n.pt) not found in current directory"
        log_info "Please download the model file and place it in this directory"
        log_info "Expected location: $(pwd)/yolo11n.pt"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

start_service() {
    log_info "Starting YOLO Stream Detection Service..."
    
    # Check if already running
    if pgrep -f "python.*main.py" > /dev/null; then
        log_warning "Service appears to be already running"
        read -p "Stop existing instance and start new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            stop_service
        else
            exit 1
        fi
    fi
    
    # Start the service
    python3 main.py &
    SERVICE_PID=$!
    
    # Wait a moment for service to start
    sleep 3
    
    # Check if service started successfully
    if kill -0 $SERVICE_PID 2>/dev/null; then
        log_success "Service started successfully (PID: $SERVICE_PID)"
        
        # Save PID to file
        echo $SERVICE_PID > service.pid
        
        log_info "Service running at: http://0.0.0.0:8000"
        log_info "Test interface available at: http://0.0.0.0:8000/test"
    else
        log_error "Failed to start service"
        exit 1
    fi
}

stop_service() {
    log_info "Stopping YOLO Stream Detection Service..."
    
    if [ -f "service.pid" ]; then
        PID=$(cat service.pid)
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            log_success "Service stopped (PID: $PID)"
            rm -f service.pid
        else
            log_warning "Service process not found"
            rm -f service.pid
        fi
    else
        # Try to find and kill the process
        pkill -f "python.*main.py"
        log_success "Service stopped"
    fi
}

show_status() {
    log_info "Checking service status..."
    
    if [ -f "service.pid" ]; then
        PID=$(cat service.pid)
        if kill -0 $PID 2>/dev/null; then
            log_success "Service is running (PID: $PID)"
            return 0
        else
            log_warning "Service PID file exists but process not running"
            rm -f service.pid
        fi
    fi
    
    if pgrep -f "python.*main.py" > /dev/null; then
        PID=$(pgrep -f "python.*main.py")
        log_success "Service is running (PID: $PID)"
        return 0
    fi
    
    log_info "Service is not running"
    return 1
}

test_service() {
    log_info "Testing service..."
    
    if ! show_status; then
        log_error "Service is not running"
        exit 1
    fi
    
    # Test basic connectivity
    log_info "Testing connectivity..."
    if curl -s "http://localhost:8000/" > /dev/null; then
        log_success "Connectivity test passed"
    else
        log_error "Connectivity test failed"
        exit 1
    fi
    
    # Test status endpoint
    log_info "Testing status endpoint..."
    if curl -s "http://localhost:8000/status" > /dev/null; then
        log_success "Status endpoint test passed"
    else
        log_error "Status endpoint test failed"
        exit 1
    fi
    
    log_success "All tests passed"
}

view_logs() {
    log_info "Recent logs (last 50 lines)..."
    
    if [ -f "service.log" ]; then
        tail -n 50 service.log
    else
        log_info "No log file found. Checking system logs..."
        journalctl -u yolo-stream --no-pager -n 50 2>/dev/null || echo "No system logs found"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop service if running
    if show_status; then
        stop_service
    fi
    
    # Remove temporary files
    rm -f service.pid
    rm -f service.log
    
    log_success "Cleanup completed"
}

# Main script
main() {
    case "${1:-help}" in
        "install")
            check_dependencies
            install_requirements
            check_model
            log_success "Installation completed"
            ;;
        "start")
            check_dependencies
            check_model
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            start_service
            ;;
        "status")
            show_status
            ;;
        "test")
            test_service
            ;;
        "monitor")
            if [ -z "$2" ]; then
                log_error "Please specify duration in seconds: ./run.sh monitor 30"
                exit 1
            fi
            python3 test_client.py --action monitor --duration "$2"
            ;;
        "logs")
            view_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"--help"|"-h")
            echo "YOLO Stream Detection Service Runner"
            echo "Usage: $0 {install|start|stop|restart|status|test|monitor|logs|cleanup|help}"
            echo ""
            echo "Commands:"
            echo "  install    - Install dependencies and check requirements"
            echo "  start      - Start the service"
            echo "  stop       - Stop the service"
            echo "  restart    - Restart the service"
            echo "  status     - Check service status"
            echo "  test       - Test service functionality"
            echo "  monitor    - Monitor stream for specified duration (e.g., monitor 30)"
            echo "  logs       - View recent logs"
            echo "  cleanup    - Clean up temporary files and stop service"
            echo "  help       - Show this help message"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"