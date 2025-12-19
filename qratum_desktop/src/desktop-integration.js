/**
 * QRATUM Desktop Edition - Dashboard Integration
 * 
 * Enhances the existing web dashboard with desktop-specific features:
 * - Native file dialogs
 * - Local configuration management
 * - Backend status monitoring
 * - Desktop UI improvements
 * 
 * @module desktop-integration
 * @license Apache-2.0
 */

(function() {
  'use strict';

  // Check if running in desktop mode
  if (!window.qratumDesktop) {
    console.warn('Not running in desktop mode, skipping desktop integration');
    return;
  }

  console.log('[Desktop] Initializing desktop integration');
  console.log('[Desktop] Platform:', window.qratumDesktop.platform);
  console.log('[Desktop] Version:', window.qratumDesktop.version);

  /**
   * Desktop configuration manager
   */
  class DesktopConfig {
    async get(key) {
      return await window.qratumDesktop.getConfig(key);
    }

    async set(key, value) {
      return await window.qratumDesktop.setConfig(key, value);
    }

    async getAll() {
      const keys = [
        'theme', 'autoStart', 'checkUpdates', 
        'backendPort', 'maxWorkers', 'useGPU'
      ];
      const config = {};
      for (const key of keys) {
        config[key] = await this.get(key);
      }
      return config;
    }
  }

  /**
   * Backend status monitor
   */
  class BackendMonitor {
    constructor() {
      this.pollInterval = 5000; // 5 seconds
      this.intervalId = null;
    }

    async checkStatus() {
      try {
        const status = await window.qratumDesktop.getBackendStatus();
        return status;
      } catch (error) {
        console.error('[Backend] Status check failed:', error);
        return { ready: false, error: error.message };
      }
    }

    startPolling(callback) {
      this.intervalId = setInterval(async () => {
        const status = await this.checkStatus();
        callback(status);
      }, this.pollInterval);
    }

    stopPolling() {
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    }

    async restart() {
      console.log('[Backend] Restarting...');
      try {
        const result = await window.qratumDesktop.restartBackend();
        if (result.success) {
          console.log('[Backend] Restart successful');
          return true;
        } else {
          console.error('[Backend] Restart failed:', result.error);
          return false;
        }
      } catch (error) {
        console.error('[Backend] Restart error:', error);
        return false;
      }
    }
  }

  /**
   * Desktop file manager
   */
  class DesktopFileManager {
    async openFile(options = {}) {
      const defaultOptions = {
        properties: ['openFile'],
        filters: [
          { name: 'All Files', extensions: ['*'] }
        ]
      };
      const result = await window.qratumDesktop.showOpenDialog({
        ...defaultOptions,
        ...options
      });
      return result;
    }

    async openDirectory() {
      const result = await window.qratumDesktop.showOpenDialog({
        properties: ['openDirectory']
      });
      return result;
    }

    async saveFile(options = {}) {
      const defaultOptions = {
        filters: [
          { name: 'All Files', extensions: ['*'] }
        ]
      };
      const result = await window.qratumDesktop.showSaveDialog({
        ...defaultOptions,
        ...options
      });
      return result;
    }
  }

  /**
   * Desktop UI enhancements
   */
  class DesktopUI {
    constructor() {
      this.config = new DesktopConfig();
      this.backend = new BackendMonitor();
      this.fileManager = new DesktopFileManager();
    }

    /**
     * Initialize desktop UI enhancements
     */
    async initialize() {
      console.log('[Desktop UI] Initializing enhancements');

      // Add desktop indicator to header
      this.addDesktopIndicator();

      // Replace file inputs with native dialogs
      this.enhanceFileInputs();

      // Monitor backend status
      this.startBackendMonitoring();

      // Apply saved theme
      await this.applySavedTheme();

      console.log('[Desktop UI] Initialization complete');
    }

    /**
     * Add desktop mode indicator to header
     */
    addDesktopIndicator() {
      const header = document.querySelector('.nav-header');
      if (!header) return;

      const indicator = document.createElement('div');
      indicator.className = 'desktop-indicator';
      indicator.innerHTML = `
        <span class="desktop-badge">🖥️ Desktop Edition</span>
      `;
      indicator.style.cssText = `
        display: flex;
        align-items: center;
        margin-left: auto;
        margin-right: 1rem;
      `;

      const badge = indicator.querySelector('.desktop-badge');
      badge.style.cssText = `
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
      `;

      const navActions = header.querySelector('.nav-actions');
      if (navActions) {
        header.insertBefore(indicator, navActions);
      }
    }

    /**
     * Enhance file inputs with native dialogs
     */
    enhanceFileInputs() {
      document.addEventListener('click', async (e) => {
        const fileButton = e.target.closest('[data-desktop-file]');
        if (!fileButton) return;

        e.preventDefault();
        const fileType = fileButton.dataset.desktopFile;

        try {
          const result = await this.fileManager.openFile({
            filters: this.getFileFilters(fileType)
          });

          if (!result.canceled && result.filePaths.length > 0) {
            const filePath = result.filePaths[0];
            console.log('[Desktop] Selected file:', filePath);

            // Trigger custom event with file path
            const event = new CustomEvent('desktopFileSelected', {
              detail: { filePath, fileType }
            });
            fileButton.dispatchEvent(event);
          }
        } catch (error) {
          console.error('[Desktop] File selection error:', error);
        }
      });
    }

    /**
     * Get file filters based on type
     */
    getFileFilters(fileType) {
      const filters = {
        'config': [
          { name: 'Configuration Files', extensions: ['json', 'yaml', 'yml', 'toml'] },
          { name: 'JSON', extensions: ['json'] },
          { name: 'YAML', extensions: ['yaml', 'yml'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        'data': [
          { name: 'Data Files', extensions: ['csv', 'json', 'parquet', 'hdf5'] },
          { name: 'CSV', extensions: ['csv'] },
          { name: 'JSON', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ],
        'results': [
          { name: 'Result Files', extensions: ['json', 'hdf5', 'npz'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      };

      return filters[fileType] || [{ name: 'All Files', extensions: ['*'] }];
    }

    /**
     * Start backend status monitoring
     */
    startBackendMonitoring() {
      this.backend.startPolling((status) => {
        this.updateBackendStatusUI(status);
      });
    }

    /**
     * Update backend status in UI
     */
    updateBackendStatusUI(status) {
      const statusDot = document.querySelector('.status-dot');
      const statusText = document.querySelector('.status-text');

      if (statusDot && statusText) {
        if (status.ready) {
          statusDot.className = 'status-dot online';
          statusText.textContent = 'Backend Online';
        } else {
          statusDot.className = 'status-dot offline';
          statusText.textContent = 'Backend Offline';
        }
      }
    }

    /**
     * Apply saved theme
     */
    async applySavedTheme() {
      const theme = await this.config.get('theme');
      if (theme) {
        document.body.setAttribute('data-theme', theme);
      }
    }
  }

  // Initialize desktop UI when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      const desktopUI = new DesktopUI();
      desktopUI.initialize();
    });
  } else {
    const desktopUI = new DesktopUI();
    desktopUI.initialize();
  }

  // Expose desktop API to window
  window.QRATUMDesktop = {
    config: new DesktopConfig(),
    backend: new BackendMonitor(),
    fileManager: new DesktopFileManager(),
    ui: new DesktopUI()
  };

  console.log('[Desktop] Integration complete - window.QRATUMDesktop available');

})();
