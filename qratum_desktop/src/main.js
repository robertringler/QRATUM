/**
 * QRATUM Desktop Edition - Main Process
 * 
 * Electron main process that manages:
 * - Application window lifecycle
 * - Python backend subprocess
 * - IPC communication bridge
 * - System tray integration
 * - Auto-updates
 * 
 * @module main
 * @license Apache-2.0
 */

const { app, BrowserWindow, ipcMain, Tray, Menu, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const Store = require('electron-store');

// Configuration store for user preferences
const store = new Store({
  defaults: {
    windowBounds: { width: 1400, height: 900 },
    theme: 'dark',
    autoStart: false,
    checkUpdates: true,
    backendPort: 8000,
    maxWorkers: 4,
    useGPU: true
  }
});

// Global references
let mainWindow = null;
let tray = null;
let pythonProcess = null;
let backendReady = false;

/**
 * Create the main application window
 */
function createWindow() {
  const bounds = store.get('windowBounds');
  
  mainWindow = new BrowserWindow({
    width: bounds.width,
    height: bounds.height,
    minWidth: 1024,
    minHeight: 768,
    title: 'QRATUM Desktop Edition',
    backgroundColor: '#0a0e1a',
    icon: path.join(__dirname, '../assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      enableRemoteModule: false,
      sandbox: true
    },
    show: false // Show when ready
  });

  // Load the dashboard
  const dashboardPath = path.join(__dirname, '../../dashboard/index.html');
  mainWindow.loadFile(dashboardPath);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
  });

  // Save window bounds on close
  mainWindow.on('close', () => {
    const bounds = mainWindow.getBounds();
    store.set('windowBounds', bounds);
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Development tools in dev mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  return mainWindow;
}

/**
 * Start Python backend subprocess
 */
function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const backendPort = store.get('backendPort');
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    const backendScript = path.join(__dirname, 'backend_server.py');

    console.log('[Backend] Starting Python backend on port', backendPort);

    pythonProcess = spawn(pythonPath, [
      backendScript,
      '--port', backendPort.toString(),
      '--desktop-mode'
    ], {
      cwd: path.join(__dirname, '../..'),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        QRATUM_DESKTOP_MODE: '1',
        QRATUM_BACKEND_PORT: backendPort.toString()
      }
    });

    // Handle stdout
    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString().trim();
      console.log('[Backend]', message);
      
      // Check if backend is ready
      if (message.includes('Backend ready') || message.includes('Uvicorn running')) {
        backendReady = true;
        resolve();
      }
    });

    // Handle stderr
    pythonProcess.stderr.on('data', (data) => {
      console.error('[Backend Error]', data.toString().trim());
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
      console.log('[Backend] Process exited with code', code);
      backendReady = false;
      
      if (code !== 0 && code !== null) {
        dialog.showErrorBox(
          'Backend Error',
          `Python backend exited unexpectedly with code ${code}. Please check logs.`
        );
      }
    });

    // Handle process error
    pythonProcess.on('error', (err) => {
      console.error('[Backend] Failed to start:', err);
      reject(err);
    });

    // Timeout if backend doesn't start in 30 seconds
    setTimeout(() => {
      if (!backendReady) {
        reject(new Error('Backend startup timeout'));
      }
    }, 30000);
  });
}

/**
 * Stop Python backend subprocess
 */
function stopPythonBackend() {
  return new Promise((resolve) => {
    if (pythonProcess) {
      console.log('[Backend] Stopping Python backend');
      
      pythonProcess.once('close', () => {
        pythonProcess = null;
        resolve();
      });

      // Send SIGTERM (graceful shutdown)
      pythonProcess.kill('SIGTERM');
      
      // Force kill after 5 seconds if still running
      setTimeout(() => {
        if (pythonProcess) {
          console.log('[Backend] Force killing backend');
          pythonProcess.kill('SIGKILL');
          pythonProcess = null;
          resolve();
        }
      }, 5000);
    } else {
      resolve();
    }
  });
}

/**
 * Create system tray icon
 */
function createTray() {
  const iconPath = path.join(__dirname, '../assets/tray-icon.png');
  tray = new Tray(iconPath);

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show QRATUM',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      }
    },
    {
      label: 'Backend Status',
      sublabel: backendReady ? 'Running' : 'Stopped',
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Settings',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.webContents.send('navigate-to', 'settings');
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Quit QRATUM',
      click: () => {
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(contextMenu);
  tray.setToolTip('QRATUM Desktop Edition');
  
  tray.on('click', () => {
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

/**
 * Setup IPC handlers
 */
function setupIPC() {
  // Get configuration
  ipcMain.handle('get-config', (event, key) => {
    return store.get(key);
  });

  // Set configuration
  ipcMain.handle('set-config', (event, key, value) => {
    store.set(key, value);
    return true;
  });

  // Get backend status
  ipcMain.handle('backend-status', () => {
    return {
      ready: backendReady,
      port: store.get('backendPort')
    };
  });

  // Restart backend
  ipcMain.handle('restart-backend', async () => {
    try {
      await stopPythonBackend();
      await startPythonBackend();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Show file picker
  ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
  });

  // Show save dialog
  ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
  });

  // Show message box
  ipcMain.handle('show-message-box', async (event, options) => {
    const result = await dialog.showMessageBox(mainWindow, options);
    return result;
  });
}

/**
 * Application ready handler
 */
app.whenReady().then(async () => {
  console.log('[App] QRATUM Desktop Edition starting...');
  console.log('[App] Version:', app.getVersion());
  console.log('[App] Platform:', process.platform);

  try {
    // Setup IPC handlers
    setupIPC();

    // Create system tray
    createTray();

    // Start Python backend
    console.log('[App] Starting backend...');
    await startPythonBackend();
    console.log('[App] Backend started successfully');

    // Create main window
    console.log('[App] Creating main window...');
    createWindow();
    console.log('[App] Application ready');

  } catch (error) {
    console.error('[App] Failed to start:', error);
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start QRATUM Desktop:\n\n${error.message}\n\nPlease check that Python is installed and try again.`
    );
    app.quit();
  }

  // macOS specific: re-create window on dock icon click
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

/**
 * Quit when all windows are closed (except macOS)
 */
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

/**
 * Cleanup before quit
 */
app.on('before-quit', async (event) => {
  event.preventDefault();
  
  console.log('[App] Shutting down...');
  
  // Stop backend
  await stopPythonBackend();
  
  // Destroy tray
  if (tray) {
    tray.destroy();
  }
  
  console.log('[App] Cleanup complete');
  app.exit(0);
});

/**
 * Handle uncaught exceptions
 */
process.on('uncaught Exception', (error) => {
  console.error('[App] Uncaught exception:', error);
  dialog.showErrorBox('Unexpected Error', error.message);
});
