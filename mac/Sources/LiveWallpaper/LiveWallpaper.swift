// LiveWallpaper — живые 4K-обои для macOS.
// Видео крутится по кругу на уровне рабочего стола (под иконками), на всех экранах и во всех Spaces.
// Видео лежат в ~/.live-wallpaper/videos (mp4 / mov / m4v).

import AppKit
import AVFoundation

let videosDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".live-wallpaper/videos", isDirectory: true)
let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]
let lastVideoKey = "lastVideo"

@MainActor
final class WallpaperController: NSObject, NSApplicationDelegate {
    private var windows: [NSWindow] = []
    private var playerLayers: [AVPlayerLayer] = []
    private let player = AVQueuePlayer()
    private var looper: AVPlayerLooper?
    private var statusItem: NSStatusItem!
    private var pausedByUser = false
    private var currentVideo: URL?

    func applicationDidFinishLaunching(_ notification: Notification) {
        try? FileManager.default.createDirectory(at: videosDir, withIntermediateDirectories: true)

        player.isMuted = true
        player.preventsDisplaySleepDuringVideoPlayback = false   // обои не должны мешать Маку засыпать
        player.actionAtItemEnd = .advance

        buildWindows()
        setupStatusItem()
        observeSystem()

        let saved = UserDefaults.standard.string(forKey: lastVideoKey).map { URL(fileURLWithPath: $0) }
        if let saved, FileManager.default.fileExists(atPath: saved.path) {
            play(saved)
        } else if let first = listVideos().first {
            play(first)
        } else {
            NSWorkspace.shared.open(videosDir)   // первый запуск: открываем папку, чтобы положить видео
        }
    }

    // MARK: - Окна на уровне рабочего стола

    private func buildWindows() {
        windows.forEach { $0.orderOut(nil) }
        windows.removeAll()
        playerLayers.removeAll()

        for screen in NSScreen.screens {
            let window = NSWindow(contentRect: screen.frame,
                                  styleMask: .borderless,
                                  backing: .buffered,
                                  defer: false,
                                  screen: screen)
            window.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.desktopWindow)))
            window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenNone]
            window.ignoresMouseEvents = true
            window.isOpaque = true
            window.backgroundColor = .black
            window.hasShadow = false
            window.isReleasedWhenClosed = false

            let view = NSView(frame: NSRect(origin: .zero, size: screen.frame.size))
            view.wantsLayer = true
            view.layer?.backgroundColor = NSColor.black.cgColor

            let layer = AVPlayerLayer(player: player)
            layer.videoGravity = .resizeAspectFill            // заполняет экран без чёрных полос
            layer.frame = view.bounds
            layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
            layer.contentsScale = screen.backingScaleFactor   // Retina: полное разрешение
            view.layer?.addSublayer(layer)

            window.contentView = view
            window.setFrame(screen.frame, display: true)
            window.orderBack(nil)

            windows.append(window)
            playerLayers.append(layer)
        }
    }

    // MARK: - Воспроизведение

    private func listVideos() -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(at: videosDir,
                                                                  includingPropertiesForKeys: nil,
                                                                  options: [.skipsHiddenFiles])) ?? []
        return files
            .filter { videoExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    private func play(_ url: URL) {
        looper?.disableLooping()
        player.removeAllItems()

        let item = AVPlayerItem(asset: AVURLAsset(url: url))
        item.preferredForwardBufferDuration = 2
        looper = AVPlayerLooper(player: player, templateItem: item)   // бесшовный повтор
        currentVideo = url
        UserDefaults.standard.set(url.path, forKey: lastVideoKey)

        if !pausedByUser { player.play() }
        rebuildMenu()
    }

    private func resumeIfNeeded() {
        guard !pausedByUser, currentVideo != nil else { return }
        player.play()
    }

    // MARK: - Системные события

    private func observeSystem() {
        let nc = NotificationCenter.default
        nc.addObserver(self, selector: #selector(screensChanged),
                       name: NSApplication.didChangeScreenParametersNotification, object: nil)

        let ws = NSWorkspace.shared.notificationCenter
        // Экран погас / Мак уснул — ставим паузу (всё равно никто не видит), потом продолжаем
        ws.addObserver(self, selector: #selector(systemSleep),
                       name: NSWorkspace.screensDidSleepNotification, object: nil)
        ws.addObserver(self, selector: #selector(systemSleep),
                       name: NSWorkspace.willSleepNotification, object: nil)
        ws.addObserver(self, selector: #selector(systemWake),
                       name: NSWorkspace.screensDidWakeNotification, object: nil)
        ws.addObserver(self, selector: #selector(systemWake),
                       name: NSWorkspace.didWakeNotification, object: nil)
        ws.addObserver(self, selector: #selector(systemWake),
                       name: NSWorkspace.sessionDidBecomeActiveNotification, object: nil)
        ws.addObserver(self, selector: #selector(spaceChanged),
                       name: NSWorkspace.activeSpaceDidChangeNotification, object: nil)
    }

    @objc private func screensChanged() {
        buildWindows()
        resumeIfNeeded()
    }

    @objc private func systemSleep() { player.pause() }

    @objc private func systemWake() {
        windows.forEach { $0.orderBack(nil) }
        resumeIfNeeded()
    }

    @objc private func spaceChanged() { resumeIfNeeded() }

    // MARK: - Меню-бар

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "play.rectangle.on.rectangle",
                                   accessibilityDescription: "Живые обои")
        }
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
        fillMenu(menu)
    }

    private func rebuildMenu() {
        if let menu = statusItem?.menu { fillMenu(menu) }
    }

    private func fillMenu(_ menu: NSMenu) {
        menu.removeAllItems()

        let videos = listVideos()
        if videos.isEmpty {
            let empty = NSMenuItem(title: "Нет видео — положи .mp4/.mov в папку", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
        } else {
            for url in videos {
                let item = NSMenuItem(title: url.deletingPathExtension().lastPathComponent,
                                      action: #selector(chooseVideo(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = url
                item.state = (url == currentVideo) ? .on : .off
                menu.addItem(item)
            }
        }

        menu.addItem(.separator())

        let pause = NSMenuItem(title: pausedByUser ? "Продолжить" : "Пауза",
                               action: #selector(togglePause), keyEquivalent: "p")
        pause.target = self
        menu.addItem(pause)

        let next = NSMenuItem(title: "Следующее видео", action: #selector(nextVideo), keyEquivalent: "n")
        next.target = self
        next.isEnabled = videos.count > 1
        menu.addItem(next)

        let folder = NSMenuItem(title: "Открыть папку с видео…", action: #selector(openFolder), keyEquivalent: "o")
        folder.target = self
        menu.addItem(folder)

        menu.addItem(.separator())

        let quit = NSMenuItem(title: "Выйти", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
    }

    @objc private func chooseVideo(_ sender: NSMenuItem) {
        guard let url = sender.representedObject as? URL else { return }
        play(url)
    }

    @objc private func togglePause() {
        pausedByUser.toggle()
        pausedByUser ? player.pause() : player.play()
        rebuildMenu()
    }

    @objc private func nextVideo() {
        let videos = listVideos()
        guard !videos.isEmpty else { return }
        let index = currentVideo.flatMap { videos.firstIndex(of: $0) } ?? -1
        play(videos[(index + 1) % videos.count])
    }

    @objc private func openFolder() { NSWorkspace.shared.open(videosDir) }

    @objc private func quit() {
        // Выход по кнопке: launchd не перезапустит (см. install.sh, KeepAlive только при падении)
        NSApp.terminate(nil)
    }
}

extension WallpaperController: NSMenuDelegate {
    // Меню пересобирается при каждом открытии — новые видео в папке видно сразу.
    // Если обои ещё пустые, а видео появилось — сразу запускаем его.
    nonisolated func menuNeedsUpdate(_ menu: NSMenu) {
        MainActor.assumeIsolated {
            if self.currentVideo == nil, let first = self.listVideos().first {
                self.play(first)
            }
            self.fillMenu(menu)
        }
    }
}

@main
struct LiveWallpaperMain {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)   // без иконки в Dock, только меню-бар
        let controller = WallpaperController()
        app.delegate = controller
        app.run()
    }
}
