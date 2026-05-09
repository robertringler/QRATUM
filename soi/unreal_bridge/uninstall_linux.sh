#!/bin/bash
# uninstall_linux.sh
# Uninstall IntentOS auto-start for Linux
# Removes the .desktop file from ~/.config/autostart

AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/IntentOS.desktop"

if [ -f "$DESKTOP_FILE" ]; then
    rm "$DESKTOP_FILE"
    echo "IntentOS auto-start uninstalled successfully."
    echo ".desktop file removed from: $DESKTOP_FILE"
else
    echo "IntentOS auto-start .desktop file not found. Nothing to uninstall."
fi