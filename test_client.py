#!/usr/bin/env python3
"""
Test client for YOLO Stream Detection Service
Provides various testing scenarios and utilities
"""

import requests
import json
import time
import argparse
from datetime import datetime
import sys

class YOLOStreamClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_connection(self):
        """Test basic connectivity to the service"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            print(f"✓ Connection successful: {response.json()}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def get_status(self):
        """Get service status"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ Failed to get status: {e}")
            return None
    
    def start_stream(self, source=0):
        """Start video stream"""
        try:
            response = self.session.post(f"{self.base_url}/stream", params={"source": source})
            response.raise_for_status()
            result = response.json()
            print(f"✓ Stream started: {result['message']}")
            return True
        except Exception as e:
            print(f"✗ Failed to start stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop video stream"""
        try:
            response = self.session.delete(f"{self.base_url}/stream")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Stream stopped: {result['message']}")
            return True
        except Exception as e:
            print(f"✗ Failed to stop stream: {e}")
            return False
    
    def get_detection_data(self):
        """Get current detection data"""
        try:
            response = self.session.get(f"{self.base_url}/data")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ Failed to get detection data: {e}")
            return None
    
    def monitor_stream(self, duration=30):
        """Monitor stream for specified duration"""
        print(f"Monitoring stream for {duration} seconds...")
        start_time = time.time()
        
        try:
            self.start_stream()
            
            while time.time() - start_time < duration:
                data = self.get_detection_data()
                if data:
                    timestamp = data.get('timestamp', 'N/A')
                    person_count = data.get('person_count', 0)
                    detections = data.get('detections', [])
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Persons: {person_count}, "
                          f"Detections: {len(detections)}")
                    
                    if detections:
                        for i, det in enumerate(detections[:3]):  # Show first 3
                            print(f"  Person {i+1}: "
                                  f"x={det['x']:.1f}, "
                                  f"y={det['y']:.1f}, "
                                  f"confidence={det['confidence']:.2f}")
                
                time.sleep(2)  # Check every 2 seconds
                
        finally:
            self.stop_stream()
    
    def stress_test(self, iterations=5):
        """Perform stress testing"""
        print(f"Starting stress test ({iterations} iterations)...")
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}:")
            
            # Start stream
            if not self.start_stream():
                continue
            
            # Wait and collect data
            time.sleep(3)
            data = self.get_detection_data()
            if data:
                print(f"  Detected {data.get('person_count', 0)} persons")
            
            # Stop stream
            self.stop_stream()
            time.sleep(1)  # Brief pause between iterations
        
        print("\nStress test completed")
    
    def save_data_to_file(self, filename=None):
        """Save detection data to file"""
        if filename is None:
            filename = f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = self.get_detection_data()
        if data:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Data saved to {filename}")
        else:
            print("✗ No data to save")

def main():
    parser = argparse.ArgumentParser(description="YOLO Stream Detection Test Client")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the service")
    parser.add_argument("--action", choices=[
        "status", "start", "stop", "data", "monitor", "stress", "save"
    ], default="status", help="Action to perform")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Duration for monitoring (seconds)")
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Iterations for stress test")
    parser.add_argument("--source", type=int, default=0, 
                       help="Video source index")
    parser.add_argument("--output", help="Output file for data saving")
    
    args = parser.parse_args()
    
    # Create client
    client = YOLOStreamClient(args.url)
    
    # Test connection first
    if not client.test_connection():
        sys.exit(1)
    
    # Perform requested action
    if args.action == "status":
        status = client.get_status()
        if status:
            print("Service Status:")
            print(f"  Service: {status.get('service', 'unknown')}")
            print(f"  Model loaded: {status.get('model_loaded', False)}")
            print(f"  Stream active: {status.get('stream_active', False)}")
            print(f"  Live data file exists: {status.get('live_data_file_exists', False)}")
    
    elif args.action == "start":
        client.start_stream(args.source)
    
    elif args.action == "stop":
        client.stop_stream()
    
    elif args.action == "data":
        data = client.get_detection_data()
        if data:
            print(json.dumps(data, indent=2))
    
    elif args.action == "monitor":
        client.monitor_stream(args.duration)
    
    elif args.action == "stress":
        client.stress_test(args.iterations)
    
    elif args.action == "save":
        client.save_data_to_file(args.output)

if __name__ == "__main__":
    main()