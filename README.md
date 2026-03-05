# YOLO Stream Detection Service

A FastAPI-based service that processes video streams in real-time using YOLOv26m to detect persons and extract their bottom-center coordinates.

## Features

- **Real-time video stream processing** with threading to prevent lag
- **YOLOv26m model integration** for person detection
- **Bottom-center coordinate extraction** for each detected person
- **Live JSON data storage** with timestamp and detection metadata
- **RESTful API endpoints** for stream control and data retrieval
- **Web-based test interface** for easy testing
- **Comprehensive error handling and logging**

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the YOLOv26m model file (`yolo26m.pt`) and place it in the project directory.

## Usage

### Running the Service

```bash
python main.py
```

The service will start on `http://0.0.0.0:8000`

### API Endpoints

#### 1. Root Endpoint
```
GET /
```
Returns service information and available endpoints.

#### 2. Status Check
```
GET /status
```
Returns the current service status including model load status and stream activity.

#### 3. Start Stream
```
POST /stream
```
Starts video stream processing. Optional query parameter `source` (default: 0 for webcam).

**Example:**
```bash
curl -X POST "http://localhost:8000/stream?source=0"
```

#### 4. Stop Stream
```
DELETE /stream
```
Stops video stream processing.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/stream"
```

#### 5. Get Detection Data
```
GET /data
```
Retrieves the current detection data from `live_data.json`.

**Response Format:**
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "person_count": 2,
  "detections": [
    {
      "x": 320.5,
      "y": 480.0,
      "confidence": 0.85,
      "timestamp": "2024-01-01T12:00:00.000000",
      "bbox": {
        "x1": 280.0,
        "y1": 150.0,
        "x2": 361.0,
        "y2": 480.0
      }
    }
  ]
}
```

#### 6. Test Interface
```
GET /test
```
Provides a web-based test interface for interacting with the service.

## Data Format

### Detection Data Structure

Each detection contains:
- `x`: Bottom-center X coordinate (float)
- `y`: Bottom-center Y coordinate (float)
- `confidence`: Detection confidence score (0.0-1.0)
- `timestamp`: ISO format timestamp
- `bbox`: Bounding box coordinates (x1, y1, x2, y2)

### Live Data File

The service continuously updates `live_data.json` with:
- Current timestamp
- Total person count
- Array of all current detections

## Architecture

### Threading Implementation

The service uses threading to ensure smooth video processing:
- **Main thread**: Handles API requests and responses
- **Processing thread**: Captures and processes video frames
- **Thread safety**: Uses locks for shared data access

### Error Handling

- Graceful model loading with fallback
- Stream connection error handling
- Frame processing error recovery
- JSON file write error management

### Performance Optimizations

- 30 FPS processing rate (configurable)
- Efficient YOLO model inference
- Minimal memory footprint
- Non-blocking API responses

## Configuration

### Environment Variables

- `MODEL_PATH`: Path to YOLO model file (default: `yolo26m.pt`)
- `LIVE_DATA_FILE`: Path to live data JSON file (default: `live_data.json`)

### Model Configuration

- Detection threshold: 0.5 confidence
- Target class: Person (class 0)
- Frame processing rate: ~30 FPS

## Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure `yolo26m.pt` file exists in the project directory
2. **Camera not accessible**: Check camera permissions and try different source index
3. **Stream not starting**: Verify camera is available and not used by another application
4. **No detections**: Check lighting conditions and camera quality

### Debug Mode

Run with debug logging:
```bash
python main.py --log-level debug
```

## Security Considerations

- CORS is enabled for all origins (configure for production)
- No authentication implemented (add as needed)
- File system access limited to project directory

## Future Enhancements

- WebSocket support for real-time data streaming
- Multiple stream support
- Detection history storage
- Configurable detection parameters
- Authentication and authorization
- Docker containerization

## License

This project is provided as-is for educational and development purposes.