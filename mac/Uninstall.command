#!/bin/bash
cd "$(dirname "$0")"
chmod +x uninstall.sh
./uninstall.sh
read -n 1 -s -r -p "Нажми любую клавишу, чтобы закрыть окно…"
