#!/usr/bin/env python3
"""
Demo script for YOLO Stream Detection Service
Shows how to use the service with sample video processing
"""

import requests
import json
import time
import cv2
import numpy as np
from pathlib import Path
import argparse

class YOLODemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def create_sample_video(self, filename="sample_video.mp4", duration=10, fps=30):
        """Create a sample video for testing"""
        print(f"Creating sample video: {filename}")
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Create frames with moving objects
        for frame_num in range(duration * fps):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add background
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Add moving rectangle (simulating a person)
            x = int((width - 100) * (frame_num / (duration * fps)))
            y = height - 200
            cv2.rectangle(frame, (x, y), (x + 80, y + 150), (0, 255, 0), 2)
            
            # Add another moving object
            x2 = int(width - 100 - (width - 100) * (frame_num / (duration * fps)))
            y2 = height - 180
            cv2.rectangle(frame, (x2, y2), (x2 + 60, y2 + 120), (255, 0, 0), 2)
            
            # Add timestamp
            timestamp = f"Frame: {frame_num}"
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✓ Sample video created: {filename}")
        return filename
    
    def simulate_camera_feed(self, video_file=None):
        """Simulate camera feed by processing video file or creating sample"""
        if video_file and Path(video_file).exists():
            return video_file
        else:
            sample_file = "sample_video.mp4"
            if not Path(sample_file).exists():
                self.create_sample_video(sample_file)
            return sample_file
    
    def test_service_endpoints(self):
        """Test all service endpoints"""
        print("\n=== Testing Service Endpoints ===")
        
        # Test root endpoint
        try:
            response = self.session.get(f"{self.base_url}/")
            print(f"✓ Root endpoint: {response.status_code}")
        except Exception as e:
            print(f"✗ Root endpoint failed: {e}")
            return False
        
        # Test status endpoint
        try:
            response = self.session.get(f"{self.base_url}/status")
            status_data = response.json()
            print(f"✓ Status endpoint: {response.status_code}")
            print(f"  Model loaded: {status_data.get('model_loaded', False)}")
            print(f"  Stream active: {status_data.get('stream_active', False)}")
        except Exception as e:
            print(f"✗ Status endpoint failed: {e}")
            return False
        
        return True
    
    def run_detection_demo(self, video_source=0, duration=15):
        """Run the detection demo"""
        print(f"\n=== Running Detection Demo ===")
        print(f"Video source: {video_source}")
        print(f"Duration: {duration} seconds")
        
        # Start stream
        try:
            response = self.session.post(f"{self.base_url}/stream", 
                                       params={"source": video_source})
            if response.status_code == 200:
                print("✓ Stream started successfully")
            else:
                print(f"✗ Failed to start stream: {response.text}")
                return
        except Exception as e:
            print(f"✗ Error starting stream: {e}")
            return
        
        # Monitor detections
        print(f"\nMonitoring detections for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    response = self.session.get(f"{self.base_url}/data")
                    if response.status_code == 200:
                        data = response.json()
                        
                        timestamp = data.get('timestamp', 'N/A')
                        person_count = data.get('person_count', 0)
                        detections = data.get('detections', [])
                        
                        print(f"\n[{timestamp}]")
                        print(f"Persons detected: {person_count}")
                        
                        if detections:
                            for i, detection in enumerate(detections):
                                x = detection.get('x', 0)
                                y = detection.get('y', 0)
                                confidence = detection.get('confidence', 0)
                                bbox = detection.get('bbox', {})
                                
                                print(f"  Person {i+1}:")
                                print(f"    Position: ({x:.1f}, {y:.1f})")
                                print(f"    Confidence: {confidence:.2f}")
                                print(f"    Bounding box: {bbox}")
                        else:
                            print("  No persons detected")
                    
                except Exception as e:
                    print(f"Error getting data: {e}")
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Stop stream
            try:
                response = self.session.delete(f"{self.base_url}/stream")
                if response.status_code == 200:
                    print("✓ Stream stopped successfully")
                else:
                    print(f"✗ Failed to stop stream: {response.text}")
            except Exception as e:
                print(f"✗ Error stopping stream: {e}")
    
    def analyze_detection_data(self, duration=10):
        """Analyze detection patterns"""
        print(f"\n=== Analyzing Detection Data ===")
        print(f"Collecting data for {duration} seconds...")
        
        detections_history = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                response = self.session.get(f"{self.base_url}/data")
                if response.status_code == 200:
                    data = response.json()
                    detections_history.append(data)
            except Exception as e:
                print(f"Error collecting data: {e}")
            
            time.sleep(1)
        
        # Analyze data
        if detections_history:
            total_detections = sum(d.get('person_count', 0) for d in detections_history)
            avg_detections = total_detections / len(detections_history)
            max_detections = max(d.get('person_count', 0) for d in detections_history)
            
            print(f"\nAnalysis Results:")
            print(f"  Data points collected: {len(detections_history)}")
            print(f"  Average persons detected: {avg_detections:.2f}")
            print(f"  Maximum persons detected: {max_detections}")
            print(f"  Total detections: {total_detections}")
            
            # Save analysis
            analysis_file = "detection_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump({
                    'analysis_timestamp': time.time(),
                    'data_points': len(detections_history),
                    'average_persons': avg_detections,
                    'max_persons': max_detections,
                    'total_detections': total_detections,
                    'raw_data': detections_history
                }, f, indent=2)
            
            print(f"  Analysis saved to: {analysis_file}")
        else:
            print("No data collected for analysis")

def main():
    parser = argparse.ArgumentParser(description="YOLO Stream Detection Demo")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the service")
    parser.add_argument("--action", choices=[
        "test", "demo", "analyze", "create-video"
    ], default="test", help="Demo action to perform")
    parser.add_argument("--video-source", type=int, default=0, 
                       help="Video source index or use sample video if -1")
    parser.add_argument("--duration", type=int, default=15, 
                       help="Duration for demo/analysis (seconds)")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = YOLODemo(args.url)
    
    # Perform requested action
    if args.action == "test":
        if demo.test_service_endpoints():
            print("\n✓ All tests passed! Service is ready.")
        else:
            print("\n✗ Some tests failed. Check service status.")
    
    elif args.action == "demo":
        video_source = args.video_source if args.video_source >= 0 else demo.simulate_camera_feed()
        demo.run_detection_demo(video_source, args.duration)
    
    elif args.action == "analyze":
        demo.analyze_detection_data(args.duration)
    
    elif args.action == "create-video":
        output_file = args.output or "sample_video.mp4"
        demo.create_sample_video(output_file, args.duration)

if __name__ == "__main__":
    main()