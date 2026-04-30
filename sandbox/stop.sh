#!/bin/bash
# QRATUM Sandbox Stop Script
# Stops all sandbox services

echo "Stopping QRATUM Sandbox services..."

# Stop QRADLE service
pkill -f "qradle_server.py" 2>/dev/null && echo "  ✓ Stopped QRADLE service" || echo "  - QRADLE not running"

# Stop QRATUM Platform
pkill -f "qratum_platform.py" 2>/dev/null && echo "  ✓ Stopped QRATUM Platform" || echo "  - Platform not running"

# Clean up temporary files
rm -f /tmp/qradle_server.py /tmp/qradle.log /tmp/platform.log 2>/dev/null

echo "Done."
