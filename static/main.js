class Dashboard {
    constructor() {
        this.isRunning = false;
        this.sessionStartTime = null;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.currentDetections = [];
        
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            cameraSelect: document.getElementById('cameraSelect'),
            confidenceSlider: document.getElementById('confidenceSlider'),
            confidenceValue: document.getElementById('confidenceValue'),
            showCoords: document.getElementById('showCoords'),
            showFPS: document.getElementById('showFPS'),
            boxColor: document.getElementById('boxColor'),
            colorValue: document.getElementById('colorValue'),
            videoStream: document.getElementById('videoStream'),
            statusBadge: document.getElementById('statusBadge'),
            fpsCounter: document.getElementById('fpsCounter'),
            personCount: document.getElementById('personCount'),
            avgConfidence: document.getElementById('avgConfidence'),
            uptime: document.getElementById('uptime'),
            detectionsList: document.getElementById('detectionsList'),
            resolutionDisplay: document.getElementById('resolutionDisplay')
        };

        this.initEventListeners();
        this.loadSettings();
    }

    initEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());
        
        this.elements.confidenceSlider.addEventListener('input', (e) => {
            this.elements.confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });

        this.elements.boxColor.addEventListener('change', (e) => {
            this.elements.colorValue.textContent = e.target.value;
        });

        this.elements.cameraSelect.addEventListener('change', () => {
            if (this.isRunning) {
                this.stop();
                setTimeout(() => this.start(), 500);
            }
        });
    }

    saveSettings() {
        const settings = {
            camera: this.elements.cameraSelect.value,
            confidence: this.elements.confidenceSlider.value,
            showCoords: this.elements.showCoords.checked,
            showFPS: this.elements.showFPS.checked,
            boxColor: this.elements.boxColor.value
        };
        localStorage.setItem('dashboardSettings', JSON.stringify(settings));
    }

    loadSettings() {
        const saved = localStorage.getItem('dashboardSettings');
        if (saved) {
            const settings = JSON.parse(saved);
            this.elements.cameraSelect.value = settings.camera || '0';
            this.elements.confidenceSlider.value = settings.confidence || '0.5';
            this.elements.confidenceValue.textContent = parseFloat(settings.confidence || '0.5').toFixed(2);
            this.elements.showCoords.checked = settings.showCoords !== false;
            this.elements.showFPS.checked = settings.showFPS !== false;
            this.elements.boxColor.value = settings.boxColor || '#00FF88';
            this.elements.colorValue.textContent = settings.boxColor || '#00FF88';
        }
    }

    start() {
        this.saveSettings();
        
        const params = new URLSearchParams({
            camera: this.elements.cameraSelect.value,
            confidence: this.elements.confidenceSlider.value,
            show_coords: this.elements.showCoords.checked ? '1' : '0',
            show_fps: this.elements.showFPS.checked ? '1' : '0',
            box_color: this.elements.boxColor.value.substring(1)
        });

        fetch(`/api/start?${params}`, { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'started') {
                    this.isRunning = true;
                    this.sessionStartTime = Date.now();
                    this.frameCount = 0;
                    
                    // Force refresh the video stream
                    this.elements.videoStream.src = '/video_feed?' + Math.random();
                    
                    this.updateUI();
                    this.startSessionTimer();
                }
            })
            .catch(err => console.error('Failed to start:', err));
    }

    stop() {
        fetch('/api/stop', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'stopped') {
                    this.isRunning = false;
                    this.updateUI();
                }
            })
            .catch(err => console.error('Failed to stop:', err));
    }

    updateUI() {
        if (this.isRunning) {
            this.elements.statusBadge.textContent = 'Running';
            this.elements.statusBadge.className = 'status-badge status-running';
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.cameraSelect.disabled = true;
        } else {
            this.elements.statusBadge.textContent = 'Idle';
            this.elements.statusBadge.className = 'status-badge status-idle';
            this.elements.personCount.textContent = '0';
            this.elements.avgConfidence.textContent = '0%';
            this.elements.uptime.textContent = '00:00';
            this.elements.detectionsList.innerHTML = '<div class="detection-empty">No detections</div>';
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.elements.cameraSelect.disabled = false;
        }
    }

    startSessionTimer() {
        if (!this.isRunning) return;
        
        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        this.elements.uptime.textContent = 
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        
        setTimeout(() => this.startSessionTimer(), 1000);
    }

    updateStats(detections) {
        this.currentDetections = detections;
        this.elements.personCount.textContent = detections.length;

        if (detections.length > 0) {
            const avgConf = (detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length * 100).toFixed(0);
            this.elements.avgConfidence.textContent = avgConf + '%';
        } else {
            this.elements.avgConfidence.textContent = '0%';
        }

        this.updateDetectionsList(detections);
    }

    updateDetectionsList(detections) {
        if (detections.length === 0) {
            this.elements.detectionsList.innerHTML = '<div class="detection-empty">No detections</div>';
            return;
        }

        this.elements.detectionsList.innerHTML = detections.map((det, i) => `
            <div class="detection-card">
                <strong>Person ${i + 1}</strong>
                <span class="detection-coord">📍 (${Math.round(det.x)}, ${Math.round(det.y)})</span>
                <span class="detection-conf">Confidence: ${(det.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    updateFPS(fps) {
        this.elements.fpsCounter.textContent = `FPS: ${fps.toFixed(1)}`;
    }

    updateResolution(width, height) {
        this.elements.resolutionDisplay.textContent = `Resolution: ${width}×${height}`;
    }
}

const dashboard = new Dashboard();

console.log('Initializing WebSocket connection...');

const videoSocket = new WebSocket(
    (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws/stats'
);

videoSocket.onopen = () => {
    console.log('✅ WebSocket connected successfully');
};

videoSocket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        console.log('📊 Stats received:', data);
        
        if (data.type === 'stats') {
            if (data.detections) {
                dashboard.updateStats(data.detections);
            }
            if (data.fps !== undefined) {
                dashboard.updateFPS(data.fps);
            }
            if (data.width && data.height) {
                dashboard.updateResolution(data.width, data.height);
            }
        }
    } catch (err) {
        console.error('Failed to parse stats:', err);
    }
};

videoSocket.onclose = () => {
    console.log('❌ Stats connection closed - reconnecting in 3 seconds...');
    setTimeout(() => {
        location.reload();
    }, 3000);
};

videoSocket.onerror = (err) => {
    console.error('❌ WebSocket error:', err);
};
