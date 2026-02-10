class MultiCameraDashboard {
    constructor() {
        this.activeCameras = new Map();
        this.availableCameras = [];
        this.peakCount = 0;
        this.chartRange = '1m';
        this.peopleHistory = [];
        this.maxHistoryPoints = 60;
        this.cameraCounter = 0;
        this.wsConnections = new Map();
        
        this.initElements();
        this.initEventListeners();
        this.initChart();
        this.loadCameras();
        this.startClock();
        this.startStatsPolling();
    }

    initElements() {
        this.elements = {
            currentTime: document.getElementById('currentTime'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            activeCameraCount: document.getElementById('activeCameraCount'),
            cameraGrid: document.getElementById('cameraGrid'),
            cameraPlaceholder: document.getElementById('cameraPlaceholder'),
            
            totalPersonCount: document.getElementById('totalPersonCount'),
            totalMaleCount: document.getElementById('totalMaleCount'),
            totalFemaleCount: document.getElementById('totalFemaleCount'),
            peakCount: document.getElementById('peakCount'),
            
            peopleChart: document.getElementById('peopleChart'),
            chartRangeBtns: document.querySelectorAll('.chart-range-btn'),
            
            activeCamerasList: document.getElementById('activeCamerasList'),
            refreshCamerasBtn: document.getElementById('refreshCamerasBtn'),
            
            cameraSelect: document.getElementById('cameraSelect'),
            cameraName: document.getElementById('cameraName'),
            confidenceSlider: document.getElementById('confidenceSlider'),
            confidenceValue: document.getElementById('confidenceValue'),
            showCoords: document.getElementById('showCoords'),
            showFPS: document.getElementById('showFPS'),
            boxColor: document.getElementById('boxColor'),
            colorValue: document.getElementById('colorValue'),
            addCameraBtn: document.getElementById('addCameraBtn'),
            stopAllBtn: document.getElementById('stopAllBtn'),
            logoutBtn: document.getElementById('logoutBtn')
        };
    }

    initEventListeners() {
        this.elements.addCameraBtn.addEventListener('click', () => this.addCamera());
        this.elements.stopAllBtn.addEventListener('click', () => this.stopAllCameras());
        this.elements.refreshCamerasBtn.addEventListener('click', () => this.loadCameras());
        
        if (this.elements.logoutBtn) {
            this.elements.logoutBtn.addEventListener('click', () => this.handleLogout());
        }
        
        this.elements.confidenceSlider.addEventListener('input', (e) => {
            this.elements.confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        this.elements.boxColor.addEventListener('change', (e) => {
            this.elements.colorValue.textContent = e.target.value.toUpperCase();
        });
        
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
                    label: 'Total People',
                    data: [],
                    borderColor: '#3B82F6',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1e2433',
                        titleColor: '#f0f2f5',
                        bodyColor: '#a0a8b8',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                        ticks: { color: '#6b7280', font: { size: 10 }, maxTicksLimit: 6 }
                    },
                    y: {
                        display: true,
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                        ticks: { color: '#6b7280', font: { size: 10 }, stepSize: 1 }
                    }
                }
            }
        });
    }

    async loadCameras() {
        try {
            const response = await fetch('/api/cameras');
            const data = await response.json();
            
            if (data.cameras && Array.isArray(data.cameras)) {
                this.availableCameras = data.cameras;
                this.updateCameraSelect();
            }
        } catch (err) {
            console.warn('Failed to load cameras:', err);
        }
        
        await this.refreshActiveCameras();
    }

    updateCameraSelect() {
        const select = this.elements.cameraSelect;
        select.innerHTML = '';
        
        const localCameras = this.availableCameras.filter(c => c.type === 'local');
        const rtspCameras = this.availableCameras.filter(c => c.type === 'rtsp');
        
        if (localCameras.length > 0) {
            const localGroup = document.createElement('optgroup');
            localGroup.label = 'Available Cameras';
            localCameras.forEach(cam => {
                const option = document.createElement('option');
                option.value = cam.id; // This is the backend ID in our case
                option.textContent = cam.name + (cam.site_name ? ` (${cam.site_name})` : '');
                option.dataset.backendId = cam.id; // Store backend ID
                localGroup.appendChild(option);
            });
            select.appendChild(localGroup);
        }
        
        if (rtspCameras.length > 0) {
            const rtspGroup = document.createElement('optgroup');
            rtspGroup.label = 'RTSP Cameras';
            rtspCameras.forEach(cam => {
                const option = document.createElement('option');
                option.value = cam.id;
                option.textContent = cam.name;
                rtspGroup.appendChild(option);
            });
            select.appendChild(rtspGroup);
        }
    }

    async addCamera() {
        const source = this.elements.cameraSelect.value;
        const name = this.elements.cameraName.value || `Camera ${this.cameraCounter + 1}`;
        const confidence = parseFloat(this.elements.confidenceSlider.value);
        const showCoords = this.elements.showCoords.checked;
        const showFps = this.elements.showFPS.checked;
        const boxColor = this.elements.boxColor.value.substring(1);
        
        this.cameraCounter++;
        const cameraId = `cam_${Date.now()}_${this.cameraCounter}`;
        
        const selectedOption = this.elements.cameraSelect.options[this.elements.cameraSelect.selectedIndex];
        const backendId = selectedOption.dataset.backendId;

        // Use backend ID as source for now, or default to 0 (webcam) if it's just an ID
        // In a real scenario, we might map this ID to a real RTSP URL if available
        // For now, we'll assume the user wants to use their local webcam (0) but map it to this backend ID
        // Or if the backend provided a source URL, we would use that.
        // Since the backend payload shows "counts" and "id", but no RTSP URL in the example,
        // we will simulate by using local webcam 0 but associating it with the backend ID.
        
        const realSource = "0"; // Force local webcam for demo purposes, mapped to selected backend ID

        const payload = {
            camera_id: cameraId,
            source: realSource,
            name: name,
            confidence: confidence,
            show_coords: showCoords,
            show_fps: showFps,
            box_color: boxColor,
            backend_camera_id: backendId ? parseInt(backendId) : null
        };
        
        try {
            this.elements.addCameraBtn.disabled = true;
            this.elements.addCameraBtn.textContent = 'Starting...';
            
            const response = await fetch('/api/camera/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (result.status === 'started' || result.status === 'already_running') {
                this.addCameraToGrid(cameraId, name, source);
                this.connectCameraWebSocket(cameraId);
                this.elements.cameraName.value = '';
            } else {
                alert(`Failed to start camera: ${result.message || 'Unknown error'}`);
            }
        } catch (err) {
            console.error('Failed to add camera:', err);
            alert('Failed to add camera. Check console for details.');
        } finally {
            this.elements.addCameraBtn.disabled = false;
            this.elements.addCameraBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"/>
                    <line x1="5" y1="12" x2="19" y2="12"/>
                </svg>
                Add Camera
            `;
        }
        
        await this.refreshActiveCameras();
    }

    addCameraToGrid(cameraId, name, source) {
        this.elements.cameraPlaceholder.style.display = 'none';
        
        const cameraCard = document.createElement('div');
        cameraCard.className = 'camera-card';
        cameraCard.id = `camera-${cameraId}`;
        cameraCard.innerHTML = `
            <div class="camera-header">
                <div class="camera-info">
                    <span class="camera-name">${name}</span>
                    <span class="camera-source">${source}</span>
                </div>
                <div class="camera-stats">
                    <span class="camera-fps" id="fps-${cameraId}">-- FPS</span>
                    <span class="camera-count" id="count-${cameraId}">0</span>
                </div>
                <button class="camera-close-btn" data-camera-id="${cameraId}" title="Stop Camera">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div class="camera-video-wrapper">
                <img class="camera-video" src="/video_feed/${cameraId}?t=${Date.now()}" alt="${name}">
                <div class="camera-overlay">
                    <div class="gender-stats">
                        <span class="male-stat" id="male-${cameraId}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="10" cy="14" r="4"/></svg>
                            0
                        </span>
                        <span class="female-stat" id="female-${cameraId}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="8" r="4"/></svg>
                            0
                        </span>
                    </div>
                </div>
            </div>
        `;
        
        const closeBtn = cameraCard.querySelector('.camera-close-btn');
        closeBtn.addEventListener('click', () => this.stopCamera(cameraId));
        
        this.elements.cameraGrid.appendChild(cameraCard);
        
        this.activeCameras.set(cameraId, {
            name: name,
            source: source,
            stats: { personCount: 0, maleCount: 0, femaleCount: 0, fps: 0 }
        });
        
        this.updateGlobalStatus();
    }

    removeCameraFromGrid(cameraId) {
        const cameraCard = document.getElementById(`camera-${cameraId}`);
        if (cameraCard) {
            cameraCard.remove();
        }
        
        this.activeCameras.delete(cameraId);
        
        if (this.wsConnections.has(cameraId)) {
            this.wsConnections.get(cameraId).close();
            this.wsConnections.delete(cameraId);
        }
        
        if (this.activeCameras.size === 0) {
            this.elements.cameraPlaceholder.style.display = 'flex';
        }
        
        this.updateGlobalStatus();
    }

    connectCameraWebSocket(cameraId) {
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${location.host}/ws/stats/${cameraId}`;
        
        const socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log(`WebSocket connected for camera ${cameraId}`);
        };
        
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.updateCameraStats(cameraId, data);
            } catch (err) {
                console.error('Failed to parse WebSocket message:', err);
            }
        };
        
        socket.onclose = () => {
            console.log(`WebSocket closed for camera ${cameraId}`);
            this.wsConnections.delete(cameraId);
        };
        
        socket.onerror = (err) => {
            console.error(`WebSocket error for camera ${cameraId}:`, err);
        };
        
        this.wsConnections.set(cameraId, socket);
    }

    updateCameraStats(cameraId, data) {
        const camera = this.activeCameras.get(cameraId);
        if (!camera) return;
        
        camera.stats = {
            personCount: data.person_count || 0,
            maleCount: data.male_count || 0,
            femaleCount: data.female_count || 0,
            fps: data.fps || 0
        };
        
        const fpsEl = document.getElementById(`fps-${cameraId}`);
        const countEl = document.getElementById(`count-${cameraId}`);
        const maleEl = document.getElementById(`male-${cameraId}`);
        const femaleEl = document.getElementById(`female-${cameraId}`);
        
        if (fpsEl) fpsEl.textContent = `${camera.stats.fps.toFixed(1)} FPS`;
        if (countEl) countEl.textContent = camera.stats.personCount;
        if (maleEl) maleEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="10" cy="14" r="4"/></svg>${camera.stats.maleCount}`;
        if (femaleEl) femaleEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="8" r="4"/></svg>${camera.stats.femaleCount}`;
        
        this.updateGlobalStats();
    }

    updateGlobalStats() {
        let totalPeople = 0;
        let totalMale = 0;
        let totalFemale = 0;
        
        this.activeCameras.forEach(camera => {
            totalPeople += camera.stats.personCount;
            totalMale += camera.stats.maleCount;
            totalFemale += camera.stats.femaleCount;
        });
        
        this.elements.totalPersonCount.textContent = totalPeople;
        this.elements.totalMaleCount.textContent = totalMale;
        this.elements.totalFemaleCount.textContent = totalFemale;
        
        if (totalPeople > this.peakCount) {
            this.peakCount = totalPeople;
            this.elements.peakCount.textContent = this.peakCount;
        }
        
        this.updateChart(totalPeople);
    }

    updateGlobalStatus() {
        const count = this.activeCameras.size;
        this.elements.activeCameraCount.textContent = count;
        
        if (count > 0) {
            this.elements.statusIndicator.className = 'status-indicator status-running';
            this.elements.statusText.textContent = 'Running';
        } else {
            this.elements.statusIndicator.className = 'status-indicator status-idle';
            this.elements.statusText.textContent = 'Idle';
        }
        
        this.updateCameraStatusList();
    }

    updateCameraStatusList() {
        const listEl = this.elements.activeCamerasList;
        
        if (this.activeCameras.size === 0) {
            listEl.innerHTML = '<div class="empty-state">No cameras active</div>';
            return;
        }
        
        listEl.innerHTML = Array.from(this.activeCameras.entries()).map(([id, cam]) => `
            <div class="camera-status-item">
                <div class="camera-status-info">
                    <span class="camera-status-name">${cam.name}</span>
                    <span class="camera-status-source">${cam.source}</span>
                </div>
                <div class="camera-status-stats">
                    <span class="stat-pill">${cam.stats.personCount} ppl</span>
                    <span class="stat-pill male">${cam.stats.maleCount} M</span>
                    <span class="stat-pill female">${cam.stats.femaleCount} F</span>
                </div>
            </div>
        `).join('');
    }

    async stopCamera(cameraId) {
        try {
            await fetch(`/api/camera/${cameraId}/stop`, { method: 'POST' });
            this.removeCameraFromGrid(cameraId);
        } catch (err) {
            console.error('Failed to stop camera:', err);
        }
        
        await this.refreshActiveCameras();
    }

    async stopAllCameras() {
        try {
            await fetch('/api/stop', { method: 'POST' });
            
            const cameraIds = Array.from(this.activeCameras.keys());
            cameraIds.forEach(id => this.removeCameraFromGrid(id));
            
        } catch (err) {
            console.error('Failed to stop all cameras:', err);
        }
        
        await this.refreshActiveCameras();
    }

    async refreshActiveCameras() {
        try {
            const response = await fetch('/api/cameras/active');
            const data = await response.json();
            
            if (data.cameras && Array.isArray(data.cameras)) {
                data.cameras.forEach(cam => {
                    if (!this.activeCameras.has(cam.camera_id) && cam.state === 'running') {
                        this.addCameraToGrid(cam.camera_id, cam.name, cam.source);
                        this.connectCameraWebSocket(cam.camera_id);
                    }
                });
            }
        } catch (err) {
            console.warn('Failed to refresh active cameras:', err);
        }
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
            this.elements.currentTime.textContent = now.toLocaleTimeString('en-US', { hour12: false });
        };
        updateClock();
        setInterval(updateClock, 1000);
    }

    startStatsPolling() {
        setInterval(() => {
            this.updateGlobalStats();
            this.updateCameraStatusList();
        }, 1000);
    }

    async handleLogout() {
        if (!confirm('Are you sure you want to logout?')) {
            return;
        }

        try {
            const response = await fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                window.location.href = '/login';
            } else {
                console.error('Logout failed');
                window.location.href = '/login';
            }
        } catch (error) {
            console.error('Error during logout:', error);
            window.location.href = '/login';
        }
    }
}

const dashboard = new MultiCameraDashboard();
