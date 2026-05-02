/**
 * QRATUM Customer Dashboard - Main Application
 * @version 2.0.0
 */

(function() {
    'use strict';

    // ========================================
    // State Management
    // ========================================
    const state = {
        currentView: 'jobs',
        jobs: [],
        selectedJob: null,
        charts: {},
        refreshIntervals: []
    };

    // ========================================
    // Initialization
    // ========================================
    
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[Dashboard] Initializing QRATUM Dashboard v2.0.0');
        
        // Initialize components
        initNavigation();
        initJobForm();
        initFileUpload();
        initModal();
        initCharts();
        initVisualization();
        
        // Connect to WebSocket
        QratumWebSocket.connect();
        
        // Setup WebSocket event handlers
        setupWebSocketHandlers();
        
        // Load initial data
        loadInitialData();
        
        // Start clock
        updateClock();
        setInterval(updateClock, 1000);
        
        // Hide loader
        setTimeout(() => {
            document.getElementById('loader').classList.add('hidden');
        }, 1500);
        
        console.log('[Dashboard] Initialization complete');
    });

    // ========================================
    // Navigation
    // ========================================
    
    function initNavigation() {
        const navTabs = document.querySelectorAll('.nav-tab');
        
        navTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const viewId = tab.dataset.view;
                switchView(viewId);
            });
        });
        
        // Handle keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.altKey) {
                switch(e.key) {
                    case '1': switchView('jobs'); break;
                    case '2': switchView('monitoring'); break;
                    case '3': switchView('results'); break;
                    case '4': switchView('resources'); break;
                }
            }
        });
    }
    
    function switchView(viewId) {
        state.currentView = viewId;
        
        // Update nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            const isActive = tab.dataset.view === viewId;
            tab.classList.toggle('active', isActive);
            tab.setAttribute('aria-selected', isActive);
        });
        
        // Update views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.toggle('active', view.id === `${viewId}-view`);
        });
        
        // View-specific initialization
        if (viewId === 'results') {
            QratumVisualization.init('three-container');
        }
        
        console.log('[Dashboard] Switched to view:', viewId);
    }

    // ========================================
    // Job Form
    // ========================================
    
    function initJobForm() {
        const form = document.getElementById('job-form');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(form);
            const jobData = {
                name: formData.get('jobName'),
                type: formData.get('simulationType'),
                backend: formData.get('backend'),
                priority: formData.get('priority'),
                shots: parseInt(formData.get('shots')) || 1000,
                precision: formData.get('precision'),
                seed: formData.get('seed') ? parseInt(formData.get('seed')) : null,
                enableTelemetry: formData.get('enableTelemetry') === 'on'
            };
            
            try {
                showToast('Submitting job...', 'info');
                
                // In demo mode, simulate job submission
                if (QratumWebSocket.isDemoMode()) {
                    const newJob = {
                        id: `job_${Date.now()}`,
                        ...jobData,
                        status: 'queued',
                        progress: 0,
                        createdAt: new Date().toISOString(),
                        eta: '-'
                    };
                    state.jobs.unshift(newJob);
                    renderJobList();
                    
                    // Simulate job starting
                    setTimeout(() => {
                        newJob.status = 'running';
                        newJob.progress = Math.floor(Math.random() * 20);
                        newJob.eta = `~${Math.floor(Math.random() * 30) + 5} min`;
                        renderJobList();
                    }, 2000);
                } else {
                    await QratumAPI.submitJob(jobData);
                }
                
                showToast(`Job "${jobData.name}" submitted successfully!`, 'success');
                form.reset();
                logToConsole(`Job submitted: ${jobData.name}`, 'success');
                
            } catch (error) {
                showToast(`Failed to submit job: ${error.message}`, 'error');
                logToConsole(`Job submission failed: ${error.message}`, 'error');
            }
        });
        
        // Template selection
        document.querySelectorAll('.template-item').forEach(item => {
            item.addEventListener('click', () => {
                const template = item.dataset.template;
                applyTemplate(template);
            });
        });
    }
    
    function applyTemplate(templateName) {
        const templates = {
            'tire-analysis': {
                name: 'Tire Compound Analysis - Batch',
                type: 'elastomeric',
                backend: 'gpu',
                priority: 'high',
                shots: 5000
            },
            'quantum-circuit': {
                name: 'Quantum Circuit Parameter Sweep',
                type: 'quantum',
                backend: 'gpu',
                priority: 'normal',
                shots: 10000
            },
            'materials-sweep': {
                name: 'Materials Parameter Sweep',
                type: 'materials',
                backend: 'hybrid',
                priority: 'normal',
                shots: 2000
            },
            'optimization': {
                name: 'QAOA Optimization Suite',
                type: 'optimization',
                backend: 'gpu',
                priority: 'high',
                shots: 1000
            }
        };
        
        const template = templates[templateName];
        if (template) {
            document.getElementById('job-name').value = template.name;
            document.getElementById('simulation-type').value = template.type;
            document.getElementById('backend').value = template.backend;
            document.getElementById('priority').value = template.priority;
            document.getElementById('shots').value = template.shots;
            
            showToast(`Template "${templateName}" applied`, 'info');
        }
    }

    // ========================================
    // File Upload
    // ========================================
    
    function initFileUpload() {
        const uploadZone = document.getElementById('file-upload-zone');
        const fileInput = document.getElementById('input-file');
        
        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }
    
    function handleFileSelection(file) {
        const validExtensions = ['.inp', '.dat', '.json', '.yaml', '.h5', '.hdf5'];
        const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!validExtensions.includes(extension)) {
            showToast('Invalid file type. Please upload .inp, .dat, .json, .yaml, or .h5 files.', 'error');
            return;
        }
        
        // Update UI to show selected file
        const uploadContent = document.querySelector('.file-upload-content');
        uploadContent.innerHTML = `
            <span class="file-icon">✓</span>
            <span class="file-text">${file.name}</span>
            <span class="file-formats">${(file.size / 1024).toFixed(1)} KB</span>
        `;
        
        showToast(`File "${file.name}" selected`, 'success');
        logToConsole(`File selected: ${file.name}`, 'info');
    }

    // ========================================
    // Job List
    // ========================================
    
    function renderJobList() {
        const jobList = document.getElementById('job-list');
        const filter = document.getElementById('queue-filter').value;
        
        let filteredJobs = state.jobs;
        if (filter !== 'all') {
            filteredJobs = state.jobs.filter(job => job.status === filter);
        }
        
        if (filteredJobs.length === 0) {
            jobList.innerHTML = `
                <div class="empty-state">
                    <p>No jobs found</p>
                </div>
            `;
            return;
        }
        
        jobList.innerHTML = filteredJobs.map(job => `
            <div class="job-item" data-job-id="${job.id}" role="listitem" tabindex="0">
                <div class="job-status-icon ${job.status}">
                    ${getStatusIcon(job.status)}
                </div>
                <div class="job-info">
                    <div class="job-name">${escapeHtml(job.name)}</div>
                    <div class="job-meta">
                        <span>${job.type}</span>
                        <span>${formatTime(job.createdAt)}</span>
                    </div>
                </div>
                ${job.status === 'running' ? `
                    <div class="job-progress">
                        <div class="progress-mini">
                            <div class="progress-mini-fill" style="width: ${job.progress}%"></div>
                        </div>
                        <span style="font-size: 0.75rem; color: var(--text-muted)">${job.progress}%</span>
                    </div>
                ` : ''}
                ${['high', 'critical'].includes(job.priority) ? `
                    <span class="priority-badge ${job.priority}">${job.priority}</span>
                ` : ''}
            </div>
        `).join('');
        
        // Add click handlers
        jobList.querySelectorAll('.job-item').forEach(item => {
            item.addEventListener('click', () => {
                const jobId = item.dataset.jobId;
                showJobDetails(jobId);
            });
            
            // Keyboard support
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const jobId = item.dataset.jobId;
                    showJobDetails(jobId);
                }
            });
        });
        
        // Update summary counts
        updateJobSummary();
    }
    
    function getStatusIcon(status) {
        const icons = {
            running: '⟳',
            queued: '⏳',
            completed: '✓',
            failed: '✗'
        };
        return icons[status] || '•';
    }
    
    function updateJobSummary() {
        const running = state.jobs.filter(j => j.status === 'running').length;
        const queued = state.jobs.filter(j => j.status === 'queued').length;
        const completed = state.jobs.filter(j => j.status === 'completed').length;
        
        document.getElementById('running-jobs').textContent = running;
        document.getElementById('queued-jobs').textContent = queued;
        document.getElementById('completed-jobs').textContent = completed;
    }
    
    function showJobDetails(jobId) {
        const job = state.jobs.find(j => j.id === jobId);
        if (!job) return;
        
        state.selectedJob = job;
        
        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = `
            <div class="job-details">
                <div class="detail-row">
                    <span class="detail-label">Job ID</span>
                    <span class="detail-value">${job.id}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Name</span>
                    <span class="detail-value">${escapeHtml(job.name)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Type</span>
                    <span class="detail-value">${job.type}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Status</span>
                    <span class="detail-value status-${job.status}">${job.status}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Progress</span>
                    <span class="detail-value">${job.progress}%</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Created</span>
                    <span class="detail-value">${formatDateTime(job.createdAt)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">ETA</span>
                    <span class="detail-value">${job.eta}</span>
                </div>
            </div>
            ${job.status === 'running' ? `
                <div class="modal-actions" style="margin-top: var(--spacing-lg);">
                    <button class="btn btn-secondary" onclick="cancelJob('${job.id}')">Cancel Job</button>
                </div>
            ` : ''}
        `;
        
        document.getElementById('modal-title').textContent = 'Job Details';
        openModal();
    }

    // ========================================
    // Charts - Crisp Standard: 1.5-2px lines, subtle grids
    // ========================================
    
    // Chart.js global defaults for scientific instrument look
    const chartDefaults = {
        font: {
            family: "'Roboto Mono', 'SF Mono', monospace",
            size: 10
        },
        color: '#606070'
    };
    
    function initCharts() {
        // Apply global defaults if Chart is available
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = chartDefaults.font.family;
            Chart.defaults.font.size = chartDefaults.font.size;
            Chart.defaults.color = chartDefaults.color;
        }
        
        // Convergence Chart
        const convergenceCtx = document.getElementById('convergence-chart');
        if (convergenceCtx) {
            state.charts.convergence = new Chart(convergenceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        borderColor: '#00f5ff',
                        backgroundColor: 'rgba(0, 245, 255, 0.05)',
                        borderWidth: 1.5,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                                lineWidth: 1
                            },
                            ticks: {
                                color: '#606070',
                                font: { size: 9 }
                            },
                            border: {
                                color: 'rgba(255,255,255,0.15)'
                            }
                        },
                        y: {
                            display: true,
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                                lineWidth: 1
                            },
                            ticks: {
                                color: '#606070',
                                font: { size: 9 }
                            },
                            border: {
                                color: 'rgba(255,255,255,0.15)'
                            }
                        }
                    }
                }
            });
        }
        
        // System Chart
        const systemCtx = document.getElementById('system-chart');
        if (systemCtx) {
            state.charts.system = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU',
                            data: [],
                            borderColor: '#00ff88',
                            borderWidth: 1.5,
                            tension: 0.3,
                            pointRadius: 0
                        },
                        {
                            label: 'Memory',
                            data: [],
                            borderColor: '#7b2cbf',
                            borderWidth: 1.5,
                            tension: 0.3,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#606070',
                                boxWidth: 12,
                                padding: 8,
                                font: { size: 9 }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            display: true,
                            max: 100,
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                                lineWidth: 1
                            },
                            ticks: {
                                color: '#606070',
                                font: { size: 9 }
                            },
                            border: {
                                color: 'rgba(255,255,255,0.15)'
                            }
                        }
                    }
                }
            });
        }
        
        // Cost Chart
        const costCtx = document.getElementById('cost-chart');
        if (costCtx) {
            state.charts.cost = new Chart(costCtx, {
                type: 'doughnut',
                data: {
                    labels: ['AWS EKS', 'Google GKE', 'Azure AKS'],
                    datasets: [{
                        data: [6200, 4100, 2150],
                        backgroundColor: [
                            '#ff9900',
                            '#4285f4',
                            '#00bcf2'
                        ],
                        borderWidth: 1,
                        borderColor: 'rgba(0,0,0,0.3)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#606070',
                                boxWidth: 10,
                                padding: 8,
                                font: { size: 9 }
                            }
                        }
                    }
                }
            });
        }
        
        // Usage Chart - Bar chart with subtle styling
        const usageCtx = document.getElementById('usage-chart');
        if (usageCtx) {
            state.charts.usage = new Chart(usageCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'GPU Hours',
                        data: [],
                        backgroundColor: 'rgba(0, 245, 255, 0.3)',
                        borderColor: 'rgba(0, 245, 255, 0.6)',
                        borderWidth: 1,
                        borderRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                                lineWidth: 1
                            },
                            ticks: {
                                color: '#606070',
                                font: { size: 8 },
                                maxRotation: 0
                            },
                            border: {
                                color: 'rgba(255,255,255,0.15)'
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255,255,255,0.05)',
                                lineWidth: 1
                            },
                            ticks: {
                                color: '#606070',
                                font: { size: 9 }
                            },
                            border: {
                                color: 'rgba(255,255,255,0.15)'
                            }
                        }
                    }
                }
            });
        }
    }
    
    function updateConvergenceChart(data) {
        if (!state.charts.convergence) return;
        
        const chart = state.charts.convergence;
        chart.data.labels = data.map(d => d.iteration);
        chart.data.datasets[0].data = data.map(d => d.loss);
        chart.update('none');
    }
    
    function updateSystemChart(cpuData, memoryData) {
        if (!state.charts.system) return;
        
        const chart = state.charts.system;
        const maxPoints = 20;
        
        chart.data.labels.push('');
        chart.data.datasets[0].data.push(cpuData);
        chart.data.datasets[1].data.push(memoryData);
        
        if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        chart.update('none');
    }
    
    function updateUsageChart(data) {
        if (!state.charts.usage) return;
        
        const chart = state.charts.usage;
        chart.data.labels = data.map(d => d.date);
        chart.data.datasets[0].data = data.map(d => d.hours || Math.random() * 100);
        chart.update('none');
    }

    // ========================================
    // Visualization
    // ========================================
    
    function initVisualization() {
        // Mode selector
        const vizMode = document.getElementById('viz-mode');
        if (vizMode) {
            vizMode.addEventListener('change', (e) => {
                QratumVisualization.setMode(e.target.value);
            });
        }
        
        // Color map selector
        const colorMap = document.getElementById('color-map');
        if (colorMap) {
            colorMap.addEventListener('change', (e) => {
                QratumVisualization.setColorMap(e.target.value);
            });
        }
        
        // Reset button
        const resetBtn = document.getElementById('viz-reset');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                QratumVisualization.resetCamera();
            });
        }
        
        // Fullscreen button
        const fullscreenBtn = document.getElementById('viz-fullscreen');
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => {
                QratumVisualization.toggleFullscreen('three-container');
            });
        }
        
        // Export buttons
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.dataset.format;
                handleExport(format);
            });
        });
    }
    
    function handleExport(format) {
        if (!state.selectedJob) {
            showToast('Please select a completed job first', 'warning');
            return;
        }
        
        showToast(`Exporting results as ${format.toUpperCase()}...`, 'info');
        
        // Simulate export in demo mode
        setTimeout(() => {
            showToast(`Export complete: ${state.selectedJob.name}.${format}`, 'success');
        }, 1500);
    }

    // ========================================
    // WebSocket Handlers
    // ========================================
    
    function setupWebSocketHandlers() {
        // GPU metrics updates
        QratumWebSocket.on('metrics:gpu', (data) => {
            updateGPUMetrics(data);
        });
        
        // Job updates
        QratumWebSocket.on('job:update', (data) => {
            const jobIndex = state.jobs.findIndex(j => j.id === data.jobId);
            if (jobIndex !== -1) {
                state.jobs[jobIndex].progress = data.progress;
                state.jobs[jobIndex].status = data.status;
                renderJobList();
            }
        });
        
        // Job completion
        QratumWebSocket.on('job:complete', (data) => {
            const jobIndex = state.jobs.findIndex(j => j.id === data.jobId);
            if (jobIndex !== -1) {
                state.jobs[jobIndex].status = 'completed';
                state.jobs[jobIndex].progress = 100;
                renderJobList();
                showToast(`Job "${state.jobs[jobIndex].name}" completed!`, 'success');
                logToConsole(`Job completed: ${state.jobs[jobIndex].name}`, 'success');
            }
        });
        
        // System metrics
        QratumWebSocket.on('metrics:system', (data) => {
            updateSystemChart(data.cpuUsage, data.memoryUsage);
        });
        
        // Alerts
        QratumWebSocket.on('alert:new', (data) => {
            addAlert(data);
            showToast(data.title, data.type);
        });
    }
    
    function updateGPUMetrics(data) {
        document.getElementById('gpu-value').textContent = `${data.utilization}%`;
        document.getElementById('vram-usage').textContent = `${data.vramUsed} / ${data.vramTotal} GB`;
        document.getElementById('gpu-temp').textContent = `${data.temperature}°C`;
        document.getElementById('gpu-power').textContent = `${data.powerDraw} W`;
        
        // Update gauge
        const gauge = document.getElementById('gpu-gauge');
        if (gauge) {
            gauge.style.setProperty('--value', data.utilization);
        }
    }
    
    function addAlert(alert) {
        const alertsList = document.getElementById('alerts-list');
        if (!alertsList) return;
        
        const alertEl = document.createElement('div');
        alertEl.className = `alert-item ${alert.type}`;
        alertEl.innerHTML = `
            <span class="alert-icon">${alert.type === 'warning' ? '⚠️' : 'ℹ️'}</span>
            <div class="alert-content">
                <span class="alert-title">${escapeHtml(alert.title)}</span>
                <span class="alert-time">Just now</span>
            </div>
        `;
        
        alertsList.insertBefore(alertEl, alertsList.firstChild);
        
        // Keep only last 10 alerts
        while (alertsList.children.length > 10) {
            alertsList.removeChild(alertsList.lastChild);
        }
    }

    // ========================================
    // Data Loading
    // ========================================
    
    async function loadInitialData() {
        try {
            // Load mock data for demo
            state.jobs = QratumAPI.mock.generateJobs();
            renderJobList();
            
            // Load convergence data
            const convergenceData = QratumAPI.mock.generateConvergenceData();
            updateConvergenceChart(convergenceData);
            
            // Load usage data
            const usageData = Array.from({ length: 30 }, (_, i) => ({
                date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0].slice(5),
                hours: Math.floor(Math.random() * 200) + 50
            }));
            updateUsageChart(usageData);
            
            // Load GPU metrics
            const gpuMetrics = QratumAPI.mock.generateGPUMetrics();
            updateGPUMetrics(gpuMetrics);
            
            // Render progress items
            renderProgressItems();
            
            logToConsole('Initial data loaded', 'success');
            
        } catch (error) {
            console.error('[Dashboard] Failed to load initial data:', error);
            logToConsole(`Failed to load data: ${error.message}`, 'error');
        }
    }
    
    function renderProgressItems() {
        const progressList = document.getElementById('progress-list');
        if (!progressList) return;
        
        const runningJobs = state.jobs.filter(j => j.status === 'running').slice(0, 3);
        
        if (runningJobs.length === 0) {
            progressList.innerHTML = '<p style="color: var(--text-muted)">No active jobs</p>';
            return;
        }
        
        progressList.innerHTML = runningJobs.map(job => `
            <div class="progress-item">
                <div class="progress-header">
                    <span>${escapeHtml(job.name)}</span>
                    <span>${job.progress}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress}%"></div>
                </div>
                <div class="progress-eta">ETA: ${job.eta}</div>
            </div>
        `).join('');
    }

    // ========================================
    // Modal
    // ========================================
    
    function initModal() {
        const modal = document.getElementById('job-modal');
        const overlay = modal.querySelector('.modal-overlay');
        const closeBtn = modal.querySelector('.modal-close');
        const closeBtn2 = modal.querySelector('.modal-close-btn');
        
        overlay.addEventListener('click', closeModal);
        closeBtn.addEventListener('click', closeModal);
        closeBtn2.addEventListener('click', closeModal);
        
        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('open')) {
                closeModal();
            }
        });
    }
    
    function openModal() {
        const modal = document.getElementById('job-modal');
        modal.classList.add('open');
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
    }
    
    function closeModal() {
        const modal = document.getElementById('job-modal');
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
    }

    // ========================================
    // Toast Notifications
    // ========================================
    
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <div class="toast-content">
                <span class="toast-message">${escapeHtml(message)}</span>
            </div>
            <button class="toast-close" aria-label="Close">&times;</button>
        `;
        
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }

    // ========================================
    // Console Logging
    // ========================================
    
    function logToConsole(message, type = 'info') {
        const console = document.getElementById('console-output');
        if (!console) return;
        
        const time = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        line.textContent = `[${time}] ${message}`;
        
        console.appendChild(line);
        console.scrollTop = console.scrollHeight;
        
        // Keep only last 100 lines
        while (console.children.length > 100) {
            console.removeChild(console.firstChild);
        }
    }
    
    // Clear console button
    document.getElementById('clear-console')?.addEventListener('click', () => {
        const console = document.getElementById('console-output');
        if (console) {
            console.innerHTML = '';
            logToConsole('Console cleared', 'info');
        }
    });

    // ========================================
    // Utility Functions
    // ========================================
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function formatTime(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }
    
    function formatDateTime(isoString) {
        return new Date(isoString).toLocaleString();
    }
    
    function updateClock() {
        const el = document.getElementById('current-time');
        if (el) {
            el.textContent = new Date().toLocaleString();
        }
    }

    // ========================================
    // Queue Filter
    // ========================================
    
    document.getElementById('queue-filter')?.addEventListener('change', () => {
        renderJobList();
    });

    // ========================================
    // Expose Global Functions
    // ========================================
    
    window.cancelJob = async function(jobId) {
        const job = state.jobs.find(j => j.id === jobId);
        if (!job) return;
        
        if (confirm(`Are you sure you want to cancel "${job.name}"?`)) {
            job.status = 'failed';
            job.progress = 0;
            renderJobList();
            closeModal();
            showToast(`Job "${job.name}" cancelled`, 'warning');
            logToConsole(`Job cancelled: ${job.name}`, 'warning');
        }
    };

})();
