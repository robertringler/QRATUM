#!/bin/bash
# install_linux.sh
# Install IntentOS auto-start for Linux (desktop environments)
# Creates a .desktop file in ~/.config/autostart

set -e

INSTALL_PATH="${1:-$(dirname "$0")/../Binaries/Linux/IntentOS.sh}"

# Check if the executable exists
if [ ! -f "$INSTALL_PATH" ]; then
    echo "Error: IntentOS executable not found at: $INSTALL_PATH"
    echo "Please ensure the game is packaged and built first."
    exit 1
fi

# Create autostart directory if it doesn't exist
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"

# Create .desktop file
DESKTOP_FILE="$AUTOSTART_DIR/IntentOS.desktop"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=IntentOS
Comment=QRATUM Sovereign Operations Interface
Exec="$INSTALL_PATH"
Terminal=false
Categories=Game;
EOF

echo "IntentOS auto-start installed successfully."
echo ".desktop file created at: $DESKTOP_FILE"
echo "The game will now launch automatically at desktop login."
echo ""
echo "Note: This requires a desktop environment that supports XDG autostart (GNOME, KDE, etc.)"
echo "For kiosk-mode or system-level auto-start, manual systemd configuration is required."