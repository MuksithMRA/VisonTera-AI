/**
 * VisionTera Professional Dashboard
 * Real-time Detection Analytics Interface
 */

class Dashboard {
    constructor() {
        this.isRunning = false;
        this.sessionStartTime = null;
        this.currentDetections = [];
        this.peakCount = 0;
        this.heatmapEnabled = true;
        this.chartRange = '1m'; // 1m, 5m, 15m
        this.peopleHistory = [];
        this.maxHistoryPoints = 60; // 60 data points for 1 minute (1 per second)
        
        this.initElements();
        this.initEventListeners();
        this.initChart();
        this.initHeatmap();
        this.loadSettings();
        this.startClock();
    }

    initElements() {
        this.elements = {
            // Header
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            currentTime: document.getElementById('currentTime'),
            
            // Video
            videoStream: document.getElementById('videoStream'),
            heatmapOverlay: document.getElementById('heatmapOverlay'),
            toggleHeatmap: document.getElementById('toggleHeatmap'),
            fpsDisplay: document.getElementById('fpsDisplay'),
            resolutionDisplay: document.getElementById('resolutionDisplay'),
            
            // Stats
            personCount: document.getElementById('personCount'),
            avgConfidence: document.getElementById('avgConfidence'),
            uptime: document.getElementById('uptime'),
            peakCount: document.getElementById('peakCount'),
            
            // Chart
            peopleChart: document.getElementById('peopleChart'),
            chartRangeBtns: document.querySelectorAll('.chart-range-btn'),
            
            // Heatmap controls
            heatmapEnabled: document.getElementById('heatmapEnabled'),
            
            // Controls
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            cameraSelect: document.getElementById('cameraSelect'),
            confidenceSlider: document.getElementById('confidenceSlider'),
            confidenceValue: document.getElementById('confidenceValue'),
            showCoords: document.getElementById('showCoords'),
            showFPS: document.getElementById('showFPS'),
            boxColor: document.getElementById('boxColor'),
            colorValue: document.getElementById('colorValue'),
            
            // Detections
            detectionsList: document.getElementById('detectionsList'),
            detectionCount: document.getElementById('detectionCount')
        };
    }

    initEventListeners() {
        // Start/Stop buttons
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());
        
        // Confidence slider
        this.elements.confidenceSlider.addEventListener('input', (e) => {
            this.elements.confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        // Color picker
        this.elements.boxColor.addEventListener('change', (e) => {
            this.elements.colorValue.textContent = e.target.value.toUpperCase();
        });
        
        // Camera change
        this.elements.cameraSelect.addEventListener('change', () => {
            if (this.isRunning) {
                this.stop();
                setTimeout(() => this.start(), 500);
            }
        });
        
        // Heatmap toggle
        this.elements.toggleHeatmap.addEventListener('click', () => {
            this.heatmapEnabled = !this.heatmapEnabled;
            this.elements.heatmapEnabled.checked = this.heatmapEnabled;
            this.elements.toggleHeatmap.classList.toggle('active', this.heatmapEnabled);
            this.updateHeatmapVisibility();
        });
        
        this.elements.heatmapEnabled.addEventListener('change', (e) => {
            this.heatmapEnabled = e.target.checked;
            this.elements.toggleHeatmap.classList.toggle('active', this.heatmapEnabled);
            this.updateHeatmapVisibility();
        });
        
        // Chart range buttons
        this.elements.chartRangeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.chartRange = e.target.dataset.range;
                this.elements.chartRangeBtns.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updateChartRange();
            });
        });
    }

    initChart() {
        const ctx = this.elements.peopleChart.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'People Count',
                    data: [],
                    borderColor: '#3B82F6',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: '#3B82F6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#1e2433',
                        titleColor: '#f0f2f5',
                        bodyColor: '#a0a8b8',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            title: (items) => items[0].label,
                            label: (item) => `${item.raw} people detected`
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#6b7280',
                            font: { size: 10 },
                            maxTicksLimit: 6
                        }
                    },
                    y: {
                        display: true,
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#6b7280',
                            font: { size: 10 },
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    initHeatmap() {
        this.heatmapData = [];
        this.heatmapDecay = 0.98; // Decay factor for heatmap persistence
        this.heatmapCanvas = this.elements.heatmapOverlay;
        this.heatmapCtx = this.heatmapCanvas.getContext('2d');
        
        // Set canvas size on load and resize
        this.resizeHeatmap();
        window.addEventListener('resize', () => this.resizeHeatmap());
    }

    resizeHeatmap() {
        const wrapper = this.heatmapCanvas.parentElement;
        this.heatmapCanvas.width = wrapper.clientWidth;
        this.heatmapCanvas.height = wrapper.clientHeight;
    }

    updateHeatmapVisibility() {
        this.heatmapCanvas.style.display = this.heatmapEnabled ? 'block' : 'none';
    }

    updateHeatmap(detections) {
        if (!this.heatmapEnabled || !this.isRunning) return;
        
        const ctx = this.heatmapCtx;
        const canvas = this.heatmapCanvas;
        
        // Apply decay to existing heatmap
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            data[i + 3] = Math.floor(data[i + 3] * this.heatmapDecay);
        }
        ctx.putImageData(imageData, 0, 0);
        
        // Add new detection points
        detections.forEach(det => {
            // Normalize detection coordinates to canvas size
            const x = (det.x / 640) * canvas.width; // Assuming 640 width frame
            const y = (det.y / 480) * canvas.height; // Assuming 480 height frame
            
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 50);
            gradient.addColorStop(0, 'rgba(239, 68, 68, 0.6)');
            gradient.addColorStop(0.4, 'rgba(245, 158, 11, 0.3)');
            gradient.addColorStop(1, 'rgba(30, 64, 175, 0)');
            
            ctx.beginPath();
            ctx.arc(x, y, 50, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        });
    }

    clearHeatmap() {
        this.heatmapCtx.clearRect(0, 0, this.heatmapCanvas.width, this.heatmapCanvas.height);
    }

    updateChart(count) {
        const now = new Date();
        const timeLabel = now.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
        
        this.peopleHistory.push({ time: timeLabel, count: count });
        
        // Limit history based on chart range
        const maxPoints = this.getMaxPointsForRange();
        if (this.peopleHistory.length > maxPoints) {
            this.peopleHistory = this.peopleHistory.slice(-maxPoints);
        }
        
        this.chart.data.labels = this.peopleHistory.map(h => h.time);
        this.chart.data.datasets[0].data = this.peopleHistory.map(h => h.count);
        this.chart.update('none');
    }

    getMaxPointsForRange() {
        switch (this.chartRange) {
            case '1m': return 60;
            case '5m': return 300;
            case '15m': return 900;
            default: return 60;
        }
    }

    updateChartRange() {
        // Keep only the points that fit the new range
        const maxPoints = this.getMaxPointsForRange();
        if (this.peopleHistory.length > maxPoints) {
            this.peopleHistory = this.peopleHistory.slice(-maxPoints);
        }
        
        this.chart.data.labels = this.peopleHistory.map(h => h.time);
        this.chart.data.datasets[0].data = this.peopleHistory.map(h => h.count);
        this.chart.update();
    }

    startClock() {
        const updateClock = () => {
            const now = new Date();
            this.elements.currentTime.textContent = now.toLocaleTimeString('en-US', { 
                hour12: false 
            });
        };
        updateClock();
        setInterval(updateClock, 1000);
    }

    saveSettings() {
        const settings = {
            camera: this.elements.cameraSelect.value,
            confidence: this.elements.confidenceSlider.value,
            showCoords: this.elements.showCoords.checked,
            showFPS: this.elements.showFPS.checked,
            boxColor: this.elements.boxColor.value,
            heatmapEnabled: this.heatmapEnabled,
            chartRange: this.chartRange
        };
        localStorage.setItem('visionteraSettings', JSON.stringify(settings));
    }

    loadSettings() {
        const saved = localStorage.getItem('visionteraSettings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                this.elements.cameraSelect.value = settings.camera || '0';
                this.elements.confidenceSlider.value = settings.confidence || '0.5';
                this.elements.confidenceValue.textContent = parseFloat(settings.confidence || '0.5').toFixed(2);
                this.elements.showCoords.checked = settings.showCoords !== false;
                this.elements.showFPS.checked = settings.showFPS !== false;
                this.elements.boxColor.value = settings.boxColor || '#3B82F6';
                this.elements.colorValue.textContent = (settings.boxColor || '#3B82F6').toUpperCase();
                this.heatmapEnabled = settings.heatmapEnabled !== false;
                this.elements.heatmapEnabled.checked = this.heatmapEnabled;
                this.elements.toggleHeatmap.classList.toggle('active', this.heatmapEnabled);
                
                if (settings.chartRange) {
                    this.chartRange = settings.chartRange;
                    this.elements.chartRangeBtns.forEach(btn => {
                        btn.classList.toggle('active', btn.dataset.range === this.chartRange);
                    });
                }
            } catch (e) {
                console.warn('Failed to load settings:', e);
            }
        }
        this.updateHeatmapVisibility();
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
                    this.peakCount = 0;
                    this.peopleHistory = [];
                    
                    // Refresh video stream
                    this.elements.videoStream.src = '/video_feed?' + Date.now();
                    
                    this.updateUI();
                    this.startSessionTimer();
                }
            })
            .catch(err => {
                console.error('Failed to start:', err);
                this.setStatus('error', 'Error');
            });
    }

    stop() {
        fetch('/api/stop', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'stopped') {
                    this.isRunning = false;
                    this.updateUI();
                    this.clearHeatmap();
                }
            })
            .catch(err => {
                console.error('Failed to stop:', err);
            });
    }

    setStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator status-${status}`;
        this.elements.statusText.textContent = text;
    }

    updateUI() {
        if (this.isRunning) {
            this.setStatus('running', 'Running');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.cameraSelect.disabled = true;
        } else {
            this.setStatus('idle', 'Idle');
            this.elements.personCount.textContent = '0';
            this.elements.avgConfidence.textContent = '0%';
            this.elements.uptime.textContent = '00:00';
            this.elements.detectionsList.innerHTML = '<div class="detection-empty">No active detections</div>';
            this.elements.detectionCount.textContent = '0';
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.elements.cameraSelect.disabled = false;
        }
    }

    startSessionTimer() {
        if (!this.isRunning) return;
        
        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const hours = Math.floor(elapsed / 3600);
        const minutes = Math.floor((elapsed % 3600) / 60);
        const seconds = elapsed % 60;
        
        let timeStr;
        if (hours > 0) {
            timeStr = `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        } else {
            timeStr = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }
        
        this.elements.uptime.textContent = timeStr;
        
        setTimeout(() => this.startSessionTimer(), 1000);
    }

    updateStats(detections) {
        this.currentDetections = detections;
        const count = detections.length;
        
        this.elements.personCount.textContent = count;
        this.elements.detectionCount.textContent = count;
        
        // Update peak count
        if (count > this.peakCount) {
            this.peakCount = count;
            this.elements.peakCount.textContent = this.peakCount;
        }

        // Calculate average confidence
        if (count > 0) {
            const avgConf = (detections.reduce((sum, d) => sum + d.confidence, 0) / count * 100).toFixed(0);
            this.elements.avgConfidence.textContent = avgConf + '%';
        } else {
            this.elements.avgConfidence.textContent = '0%';
        }

        this.updateDetectionsList(detections);
        this.updateHeatmap(detections);
        this.updateChart(count);
    }

    updateDetectionsList(detections) {
        if (detections.length === 0) {
            this.elements.detectionsList.innerHTML = '<div class="detection-empty">No active detections</div>';
            return;
        }

        this.elements.detectionsList.innerHTML = detections.map((det, i) => `
            <div class="detection-card">
                <strong>Person ${i + 1}</strong>
                <span class="detection-coord">(${Math.round(det.x)}, ${Math.round(det.y)})</span>
                <span class="detection-conf">${(det.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    updateFPS(fps) {
        this.elements.fpsDisplay.textContent = `${fps.toFixed(1)} FPS`;
    }

    updateResolution(width, height) {
        this.elements.resolutionDisplay.textContent = `${width}×${height}`;
    }
}

// Initialize Dashboard
const dashboard = new Dashboard();

// WebSocket Connection
console.log('Initializing WebSocket connection...');

const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${location.host}/ws/stats`;

let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 3000;

function connectWebSocket() {
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log('✅ WebSocket connected');
        reconnectAttempts = 0;
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
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
            console.error('Failed to parse WebSocket message:', err);
        }
    };

    socket.onclose = () => {
        console.log('WebSocket closed');
        
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Reconnecting in ${reconnectDelay/1000}s (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`);
            setTimeout(connectWebSocket, reconnectDelay);
        } else {
            console.error('Max reconnect attempts reached');
        }
    };

    socket.onerror = (err) => {
        console.error('WebSocket error:', err);
    };

    return socket;
}

const videoSocket = connectWebSocket();
