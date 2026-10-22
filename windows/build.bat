@echo off
rem Сборка LiveWallpaper.exe компилятором, который уже есть в Windows (.NET Framework 4)
chcp 65001 >nul
cd /d "%~dp0"
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
  echo Не найден компилятор .NET Framework 4.
  pause
  exit /b 1
)
"%CSC%" /nologo /target:winexe /optimize+ /win32icon:icon.ico /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Core.dll /out:LiveWallpaper.exe LiveWallpaper.cs
if errorlevel 1 (
  echo Ошибка сборки.
  pause
  exit /b 1
)
echo Готово: LiveWallpaper.exe
pause
