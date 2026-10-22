#!/bin/bash
# Сборка и установка LiveWallpaper. Запуск: ./install.sh
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="LiveWallpaper"
APP_DIR="$HOME/Applications/$APP_NAME.app"
DATA_DIR="$HOME/.live-wallpaper"
LABEL="com.livewallpaper.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if ! xcrun --find swiftc >/dev/null 2>&1; then
  echo "Нужны бесплатные инструменты разработчика Apple (Command Line Tools, ~1 ГБ)."
  echo "Сейчас откроется окно установки — нажми «Установить», дождись конца"
  echo "и запусти Install.command ещё раз."
  xcode-select --install 2>/dev/null || true
  exit 1
fi

echo "→ Собираю…"
mkdir -p build
swiftc -O -parse-as-library \
  -target "$(uname -m)-apple-macos13.0" \
  Sources/LiveWallpaper/LiveWallpaper.swift \
  -o "build/$APP_NAME"

echo "→ Упаковываю в $APP_DIR"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/uz.umid.livewallpaper" 2>/dev/null || true
pkill -x "$APP_NAME" 2>/dev/null || true

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
cp "build/$APP_NAME" "$APP_DIR/Contents/MacOS/$APP_NAME"
cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$LABEL</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
codesign --force --sign - "$APP_DIR" >/dev/null 2>&1 || true

mkdir -p "$DATA_DIR/videos"
if [ -d videos ]; then
  cp -n videos/*.mov videos/*.mp4 videos/*.m4v "$DATA_DIR/videos/" 2>/dev/null || true
fi

echo "→ Ставлю автозапуск"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$APP_DIR/Contents/MacOS/$APP_NAME</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardErrorPath</key><string>$DATA_DIR/error.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo ""
echo "✓ Готово. Иконка обоев — в меню-баре сверху справа."
echo "  Видео клади сюда: $DATA_DIR/videos  (mp4 / mov / m4v, лучше 4K HEVC)"
