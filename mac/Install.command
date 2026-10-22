#!/bin/bash
cd "$(dirname "$0")"
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x install.sh uninstall.sh
./install.sh
echo ""
read -n 1 -s -r -p "Нажми любую клавишу, чтобы закрыть окно…"
