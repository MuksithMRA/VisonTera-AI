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
        this.loadDetectionModels();
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
            totalFlow: document.getElementById('totalFlow'),
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
            logoutBtn: document.getElementById('logoutBtn'),
            countingLine: document.getElementById('countingLine'),
            
            // Modal Elements
            modal: document.getElementById('customModal'),
            modalIcon: document.getElementById('modalIcon'),
            modalTitle: document.getElementById('modalTitle'),
            modalMessage: document.getElementById('modalMessage'),
            modalCloseBtn: document.getElementById('modalCloseBtn'),
            
            // Model Switcher Elements
            modelSelect: document.getElementById('modelSelect'),
            switchModelBtn: document.getElementById('switchModelBtn'),
            currentModelName: document.getElementById('currentModelName')
        };
    }

    initEventListeners() {
        this.elements.addCameraBtn.addEventListener('click', () => this.addCamera());
        this.elements.stopAllBtn.addEventListener('click', () => this.stopAllCameras());
        this.elements.refreshCamerasBtn.addEventListener('click', () => this.loadCameras());
        
        if (this.elements.logoutBtn) {
            this.elements.logoutBtn.addEventListener('click', () => this.handleLogout());
        }
        
        if (this.elements.modalCloseBtn) {
            this.elements.modalCloseBtn.addEventListener('click', () => this.closeModal());
        }
        
        if (this.elements.modal) {
            this.elements.modal.addEventListener('click', (e) => {
                if (e.target === this.elements.modal) this.closeModal();
            });
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
        
        if (this.elements.switchModelBtn) {
            this.elements.switchModelBtn.addEventListener('click', () => this.switchModel());
        }
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
        
        if (this.availableCameras.length > 0) {
            const group = document.createElement('optgroup');
            group.label = 'Available Cameras';
            
            this.availableCameras.forEach(cam => {
                const option = document.createElement('option');
                option.value = cam.id;
                option.textContent = cam.name + (cam.site_name ? ` (${cam.site_name})` : '');
                option.dataset.backendId = cam.id;
                option.dataset.backendUrl = cam.url || '';
                group.appendChild(option);
            });
            
            select.appendChild(group);
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

        // Check if the selected backend camera has a specific URL
        const backendUrl = selectedOption.dataset.backendUrl;
        
        // If we have a backend RTSP/HTTP URL, use that. 
        // Otherwise, fallback to "0" for the local webcam.
        let realSource = "1";
        if (backendUrl && backendUrl !== "undefined" && backendUrl !== "null" && backendUrl.trim() !== "") {
            realSource = backendUrl;
        }

        const payload = {
            camera_id: cameraId,
            source: realSource,
            name: name,
            confidence: confidence,
            show_coords: showCoords,
            show_fps: showFps,
            box_color: boxColor,
            backend_camera_id: backendId ? parseInt(backendId) : null,
            counting_line: this.parseCountingLine(this.elements.countingLine.value)
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
                this.showModal('Error Starting Camera', `Failed to start camera: ${result.message || 'Unknown error'}`);
            }
        } catch (err) {
            console.error('Failed to add camera:', err);
            this.showModal('System Error', 'Failed to add camera. Check console for details.');
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

    parseCountingLine(value) {
        if (!value || !value.trim()) return null;
        try {
            const parts = value.split(',').map(p => parseFloat(p.trim()));
            if (parts.length === 4) {
                return [[parts[0], parts[1]], [parts[2], parts[3]]];
            }
        } catch (e) {
            console.error('Invalid counting line format:', e);
        }
        return null;
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
                    <span class="camera-count" id="count-${cameraId}" title="In Frame">0</span>
                    <span class="camera-cross-count" id="cross-${cameraId}" title="Total Flow">0</span>
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
            fps: data.fps || 0,
            detections: data.detections || [],
            deduplicated: data.deduplicated || null,
            crossCount: data.cross_count || 0
        };
        
        const fpsEl = document.getElementById(`fps-${cameraId}`);
        const countEl = document.getElementById(`count-${cameraId}`);
        const crossEl = document.getElementById(`cross-${cameraId}`);
        const maleEl = document.getElementById(`male-${cameraId}`);
        const femaleEl = document.getElementById(`female-${cameraId}`);
        
        if (fpsEl) fpsEl.textContent = `${camera.stats.fps.toFixed(1)} FPS`;
        if (countEl) countEl.textContent = camera.stats.personCount;
        if (crossEl) crossEl.textContent = `Flow: ${camera.stats.crossCount}`;
        if (maleEl) maleEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="10" cy="14" r="4"/></svg>${camera.stats.maleCount}`;
        if (femaleEl) femaleEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="8" r="4"/></svg>${camera.stats.femaleCount}`;
        
        this.updateGlobalStats();
    }

    updateGlobalStats() {
        // ── Cross-Camera Deduplication ──
        // Priority:
        //   1. Backend deduplicated counts (authoritative, merge-aware)
        //   2. Unique global_id counting from detections (fallback)
        //   3. Naive per-camera sum (last resort)
        
        let totalPeople, totalMale, totalFemale;
        let resolved = false;
        
        // ── Priority 1: Use backend deduplicated counts ──
        // The backend's Re-ID manager maintains merged, authoritative counts.
        // This is the most accurate source since it reflects real-time merges.
        let dedupData = null;
        this.activeCameras.forEach(camera => {
            if (camera.stats.deduplicated && camera.stats.deduplicated.total !== undefined) {
                dedupData = camera.stats.deduplicated;
            }
        });
        
        if (dedupData && dedupData.total > 0) {
            totalPeople = dedupData.total;
            totalMale = dedupData.male || 0;
            totalFemale = dedupData.female || 0;
            resolved = true;
        }
        
        // ── Priority 2: Count unique global_ids from detections ──
        if (!resolved) {
            const globalPersons = new Map();
            let hasGlobalIds = false;
            
            this.activeCameras.forEach(camera => {
                const detections = camera.stats.detections || [];
                detections.forEach(det => {
                    const gid = det.global_id;
                    if (gid !== undefined && gid !== null && gid !== -1) {
                        hasGlobalIds = true;
                        if (!globalPersons.has(gid)) {
                            globalPersons.set(gid, {
                                gender: det.gender || 'Person'
                            });
                        }
                    }
                });
            });
            
            if (hasGlobalIds && globalPersons.size > 0) {
                totalPeople = globalPersons.size;
                totalMale = 0;
                totalFemale = 0;
                globalPersons.forEach(person => {
                    if (person.gender === 'Male') totalMale++;
                    else if (person.gender === 'Female') totalFemale++;
                });
                resolved = true;
            }
        }
        
        // ── Priority 3: Naive per-camera sum ──
        if (!resolved) {
            totalPeople = 0;
            totalMale = 0;
            totalFemale = 0;
            this.activeCameras.forEach(camera => {
                totalPeople += camera.stats.personCount;
                totalMale += camera.stats.maleCount;
                totalFemale += camera.stats.femaleCount;
            });
        }
        
        this.elements.totalPersonCount.textContent = totalPeople;
        this.elements.totalMaleCount.textContent = totalMale;
        this.elements.totalFemaleCount.textContent = totalFemale;

        // Calculate Total Flow (Crossings)
        let totalFlow = 0;
        this.activeCameras.forEach(camera => {
            totalFlow += (camera.stats.crossCount || 0);
        });
        if (this.elements.totalFlow) {
            this.elements.totalFlow.textContent = totalFlow;
        }
        
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

    handleLogout() {
        if (!confirm('Are you sure you want to logout?')) {
            return;
        }

        try {
            fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            }).finally(() => {
                 window.location.href = '/login';
            });
        } catch (error) {
            console.error('Error during logout:', error);
            window.location.href = '/login';
        }
    }

    showModal(title, message, isError = true) {
        if (this.elements.modalTitle) this.elements.modalTitle.textContent = title;
        if (this.elements.modalMessage) this.elements.modalMessage.textContent = message;
        
        if (this.elements.modalIcon) {
            if (isError) {
                this.elements.modalIcon.className = 'modal-icon error';
                this.elements.modalIcon.innerHTML = `
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                `;
            } else {
                this.elements.modalIcon.className = 'modal-icon info';
                this.elements.modalIcon.innerHTML = `
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                `;
            }
        }
        
        if (this.elements.modal) this.elements.modal.classList.add('active');
    }

    closeModal() {
        if (this.elements.modal) this.elements.modal.classList.remove('active');
    }

    async loadDetectionModels() {
        try {
            const response = await fetch('/api/models/detection');
            const data = await response.json();
            
            if (data.models && Array.isArray(data.models)) {
                const select = this.elements.modelSelect;
                select.innerHTML = '';
                
                // Group models
                const trained = data.models.filter(m => m.type === 'trained');
                const base = data.models.filter(m => m.type === 'base');
                
                if (trained.length > 0) {
                    const group = document.createElement('optgroup');
                    group.label = '🎯 Trained Models';
                    trained.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.path;
                        opt.textContent = m.name + (m.active ? ' ✅' : '');
                        opt.selected = m.active;
                        group.appendChild(opt);
                    });
                    select.appendChild(group);
                }
                
                if (base.length > 0) {
                    const group = document.createElement('optgroup');
                    group.label = '📦 Base Models';
                    base.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.path;
                        opt.textContent = m.name + (m.active ? ' ✅' : '');
                        opt.selected = m.active;
                        group.appendChild(opt);
                    });
                    select.appendChild(group);
                }
                
                // Update indicator
                const active = data.models.find(m => m.active);
                if (active && this.elements.currentModelName) {
                    this.elements.currentModelName.textContent = active.name;
                }
            }
        } catch (err) {
            console.warn('Failed to load detection models:', err);
            if (this.elements.currentModelName) {
                this.elements.currentModelName.textContent = 'Error loading';
            }
        }
    }

    async switchModel() {
        const select = this.elements.modelSelect;
        const modelPath = select.value;
        
        if (!modelPath) {
            this.showModal('No Model Selected', 'Please select a detection model to switch to.');
            return;
        }
        
        const btn = this.elements.switchModelBtn;
        try {
            btn.disabled = true;
            btn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
                    <polyline points="23 4 23 10 17 10"/>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                </svg>
                Switching...
            `;
            
            const response = await fetch('/api/models/detection/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelPath })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showModal('Model Switched', `Now using: ${result.model}\nDevice: ${result.device}`, false);
                await this.loadDetectionModels();
            } else {
                this.showModal('Switch Failed', result.message || 'Unknown error');
            }
        } catch (err) {
            console.error('Failed to switch model:', err);
            this.showModal('System Error', 'Failed to switch model. Check console.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"/>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                </svg>
                Switch
            `;
        }
    }
}

const dashboard = new MultiCameraDashboard();
