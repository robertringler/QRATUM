/**
 * QRATUM Desktop Edition - Preload Script
 * 
 * Secure bridge between renderer and main process.
 * Exposes only whitelisted APIs to the web page.
 * 
 * @module preload
 * @license Apache-2.0
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('qratumDesktop', {
  
  // Application info
  platform: process.platform,
  version: process.env.npm_package_version || '2.0.0',
  
  // Configuration management
  getConfig: (key) => ipcRenderer.invoke('get-config', key),
  setConfig: (key, value) => ipcRenderer.invoke('set-config', key, value),
  
  // Backend control
  getBackendStatus: () => ipcRenderer.invoke('backend-status'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  
  // File dialogs
  showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
  
  // Navigation events
  onNavigate: (callback) => {
    ipcRenderer.on('navigate-to', (event, destination) => callback(destination));
  },
  
  // Desktop mode flag
  isDesktopMode: true
});

// Log desktop mode initialization
console.log('[Preload] QRATUM Desktop Edition - Context bridge initialized');
console.log('[Preload] Platform:', process.platform);
