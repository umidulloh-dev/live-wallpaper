#!/bin/bash
# Удаление LiveWallpaper. Видео в ~/.live-wallpaper/videos остаются.
LABEL="com.livewallpaper.app"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
pkill -x LiveWallpaper 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
rm -rf "$HOME/Applications/LiveWallpaper.app"
echo "✓ Удалено. Видео остались в ~/.live-wallpaper/videos"
