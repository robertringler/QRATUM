#!/bin/bash
# package_linux.sh
# Package IntentOS for Linux distribution
# Creates a tar.gz archive and optional systemd unit

set -e

PACKAGE_NAME="IntentOS_Linux"
PACKAGE_DIR="PackagedGame/Linux"
OUTPUT_DIR="${1:-.}"
SYSTEMD_INSTALL="${2:-false}"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "Error: Packaged game not found at: $PACKAGE_DIR"
    echo "Please build and package the game first."
    exit 1
fi

# Create package directory
PACKAGE_ROOT="$OUTPUT_DIR/$PACKAGE_NAME"
mkdir -p "$PACKAGE_ROOT"

# Copy game files
cp -r "$PACKAGE_DIR"/* "$PACKAGE_ROOT/"

# Copy install scripts
cp install_linux.sh uninstall_linux.sh "$PACKAGE_ROOT/"

# Create systemd unit file (optional)
if [ "$SYSTEMD_INSTALL" = "true" ]; then
    cat > "$PACKAGE_ROOT/intentos.service" << EOF
[Unit]
Description=IntentOS Sovereign Operations Interface
After=graphical.target
Wants=graphical.target

[Service]
Type=simple
User=%i
Environment=DISPLAY=:0
ExecStart=$PACKAGE_ROOT/IntentOS.sh
Restart=always
RestartSec=5

[Install]
WantedBy=graphical.target
EOF

    echo "Systemd unit file created: $PACKAGE_ROOT/intentos.service"
    echo "To install system-wide:"
    echo "  sudo cp intentos.service /etc/systemd/system/"
    echo "  sudo systemctl enable intentos@\$USER.service"
    echo "  sudo systemctl start intentos@\$USER.service"
fi

# Create archive
ARCHIVE_NAME="$PACKAGE_NAME.tar.gz"
cd "$OUTPUT_DIR"
tar -czf "$ARCHIVE_NAME" "$PACKAGE_NAME"

echo "Package created: $OUTPUT_DIR/$ARCHIVE_NAME"
echo "To install:"
echo "  tar -xzf $ARCHIVE_NAME"
echo "  cd $PACKAGE_NAME"
echo "  ./install_linux.sh"

# Cleanup
rm -rf "$PACKAGE_ROOT"