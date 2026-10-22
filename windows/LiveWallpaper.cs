// LiveWallpaper для Windows — живые 4K-обои под иконками рабочего стола.
// Иконка в трее: выбор видео, пауза, папка с видео, автозапуск, выход.
// Сборка: build.bat (компилятор уже есть в каждой Windows 10/11, ставить ничего не нужно).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace LiveWallpaper
{
    static class Program
    {
        public static readonly string InstallDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LiveWallpaper");
        public static readonly string VideosDir = Path.Combine(InstallDir, "videos");
        public static readonly string SettingsFile = Path.Combine(InstallDir, "settings.txt");
        public static readonly string[] VideoExt = { ".mp4", ".mov", ".m4v", ".wmv", ".avi" };
        const string RunKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
        const string AppName = "LiveWallpaper";

        [STAThread]
        static void Main()
        {
            try { Native.SetProcessDPIAware(); } catch { }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            bool created;
            using (var mutex = new Mutex(true, "LiveWallpaper_single_instance", out created))
            {
                string exeDir = Path.GetDirectoryName(Application.ExecutablePath);
                bool installed = string.Equals(Path.GetFullPath(exeDir).TrimEnd('\\'),
                                               Path.GetFullPath(InstallDir).TrimEnd('\\'),
                                               StringComparison.OrdinalIgnoreCase);
                if (!created)
                {
                    MessageBox.Show(installed
                        ? "LiveWallpaper уже запущен — иконка в трее, справа внизу (может прятаться под стрелкой ^)."
                        : "LiveWallpaper уже запущен. Чтобы переустановить или обновить, сначала закрой его:\nиконка в трее → «Выйти».",
                        AppName, MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }
                if (!installed)
                {
                    mutex.ReleaseMutex();       // иначе установленная копия решит, что уже запущена
                    Install(exeDir);
                    return;
                }
                Directory.CreateDirectory(VideosDir);
                Application.Run(new TrayApp());
            }
        }

        // Первый запуск из «Загрузок»: копируем себя и видео в %LOCALAPPDATA%\LiveWallpaper и ставим автозапуск
        static void Install(string fromDir)
        {
            try
            {
                Directory.CreateDirectory(VideosDir);
                string target = Path.Combine(InstallDir, "LiveWallpaper.exe");
                File.Copy(Application.ExecutablePath, target, true);
                string srcVideos = Path.Combine(fromDir, "videos");
                int copied = 0;
                if (Directory.Exists(srcVideos))
                {
                    foreach (string f in Directory.GetFiles(srcVideos))
                    {
                        if (!IsVideo(f)) continue;
                        string dst = Path.Combine(VideosDir, Path.GetFileName(f));
                        if (!File.Exists(dst)) { File.Copy(f, dst); copied++; }
                    }
                }
                SetAutostart(true, target);
                Process.Start(target);
                MessageBox.Show("Готово! Живые обои установлены" + (copied > 0 ? " (видео: " + copied + ")" : "") + ".\n\n" +
                                "Иконка — в трее справа внизу (может прятаться под стрелкой ^).\n" +
                                "Через неё выбираешь обои, ставишь паузу и добавляешь свои видео.\n\n" +
                                "Обои будут запускаться сами при входе в Windows.",
                                AppName, MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Не получилось установить: " + ex.Message, AppName, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public static bool IsVideo(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return VideoExt.Contains(ext);
        }

        public static bool GetAutostart()
        {
            using (var k = Registry.CurrentUser.OpenSubKey(RunKey))
                return k != null && k.GetValue(AppName) != null;
        }

        public static void SetAutostart(bool on, string exe)
        {
            using (var k = Registry.CurrentUser.CreateSubKey(RunKey))
            {
                if (on) k.SetValue(AppName, "\"" + exe + "\"");
                else if (k.GetValue(AppName) != null) k.DeleteValue(AppName);
            }
        }
    }

    // ------------------------------------------------------------------ трей + управление
    class TrayApp : ApplicationContext
    {
        readonly NotifyIcon tray;
        readonly ContextMenuStrip menu;
        readonly MessageWindow messages;
        readonly System.Windows.Forms.Timer layoutTimer;
        readonly List<WallpaperWindow> windows = new List<WallpaperWindow>();
        string current;
        bool pausedByUser;
        bool pausedBySystem;

        public TrayApp()
        {
            menu = new ContextMenuStrip();
            menu.Opening += delegate { BuildMenu(); };

            Icon icon;
            try { icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); }
            catch { icon = SystemIcons.Application; }
            tray = new NotifyIcon { Icon = icon, Text = "Живые обои", ContextMenuStrip = menu, Visible = true };
            tray.MouseClick += delegate(object s, MouseEventArgs e)
            {
                if (e.Button == MouseButtons.Left)
                {
                    // показать меню и по левому клику
                    MethodInfo mi = typeof(NotifyIcon).GetMethod("ShowContextMenu", BindingFlags.Instance | BindingFlags.NonPublic);
                    if (mi != null) mi.Invoke(tray, null);
                }
            };

            messages = new MessageWindow();
            messages.ExplorerRestarted += delegate { Rebuild(); };
            SystemEvents.DisplaySettingsChanged += delegate { Rebuild(); };
            SystemEvents.SessionSwitch += OnSessionSwitch;
            SystemEvents.PowerModeChanged += OnPowerMode;

            // подгоняем «обрезку» видео, когда стал известен его размер
            layoutTimer = new System.Windows.Forms.Timer { Interval = 500 };
            layoutTimer.Tick += delegate { foreach (var w in windows) w.UpdateLayoutIfReady(); };
            layoutTimer.Start();

            current = LoadLast();
            if (current == null || !File.Exists(current))
                current = ListVideos().FirstOrDefault();
            Rebuild();
            if (current == null)
            {
                tray.ShowBalloonTip(5000, "Живые обои", "Положи видео (.mp4) в папку — сейчас откроется.", ToolTipIcon.Info);
                OpenFolder();
            }
        }

        static List<string> ListVideos()
        {
            if (!Directory.Exists(Program.VideosDir)) return new List<string>();
            return Directory.GetFiles(Program.VideosDir)
                .Where(Program.IsVideo)
                .OrderBy(f => Path.GetFileName(f), StringComparer.CurrentCultureIgnoreCase)
                .ToList();
        }

        void Rebuild()
        {
            foreach (var w in windows) w.Dispose();
            windows.Clear();
            if (current == null) return;

            IntPtr parent, insertAfter;
            if (!Desktop.FindHost(out parent, out insertAfter)) return;
            Rectangle virt = SystemInformation.VirtualScreen;
            foreach (Screen s in Screen.AllScreens)
            {
                var w = new WallpaperWindow();
                w.Attach(parent, insertAfter, new Rectangle(s.Bounds.X - virt.X, s.Bounds.Y - virt.Y, s.Bounds.Width, s.Bounds.Height));
                w.Play(current);
                if (pausedByUser || pausedBySystem) w.Pause();
                windows.Add(w);
            }
        }

        void Play(string path)
        {
            current = path;
            SaveLast(path);
            if (windows.Count == 0) Rebuild();
            else foreach (var w in windows) w.Play(path);
            if (pausedByUser || pausedBySystem) foreach (var w in windows) w.Pause();
        }

        void ApplyPause()
        {
            bool p = pausedByUser || pausedBySystem;
            foreach (var w in windows) { if (p) w.Pause(); else w.Resume(); }
        }

        void OnSessionSwitch(object s, SessionSwitchEventArgs e)
        {
            if (e.Reason == SessionSwitchReason.SessionLock || e.Reason == SessionSwitchReason.ConsoleDisconnect) pausedBySystem = true;
            if (e.Reason == SessionSwitchReason.SessionUnlock || e.Reason == SessionSwitchReason.ConsoleConnect) pausedBySystem = false;
            ApplyPause();
        }

        void OnPowerMode(object s, PowerModeChangedEventArgs e)
        {
            if (e.Mode == PowerModes.Suspend) { pausedBySystem = true; ApplyPause(); }
            if (e.Mode == PowerModes.Resume) { pausedBySystem = false; Rebuild(); }
        }

        void BuildMenu()
        {
            menu.Items.Clear();
            var videos = ListVideos();
            if (videos.Count == 0)
            {
                menu.Items.Add(new ToolStripMenuItem("Нет видео — положи .mp4 в папку") { Enabled = false });
            }
            foreach (string v in videos)
            {
                string path = v;
                var item = new ToolStripMenuItem(Path.GetFileNameWithoutExtension(v));
                item.Checked = string.Equals(v, current, StringComparison.OrdinalIgnoreCase);
                item.Click += delegate { Play(path); };
                menu.Items.Add(item);
            }
            menu.Items.Add(new ToolStripSeparator());

            var pause = new ToolStripMenuItem(pausedByUser ? "Продолжить" : "Пауза");
            pause.Click += delegate { pausedByUser = !pausedByUser; ApplyPause(); };
            menu.Items.Add(pause);

            var next = new ToolStripMenuItem("Следующее видео") { Enabled = videos.Count > 1 };
            next.Click += delegate
            {
                int i = videos.FindIndex(x => string.Equals(x, current, StringComparison.OrdinalIgnoreCase));
                Play(videos[(i + 1) % videos.Count]);
            };
            menu.Items.Add(next);

            var folder = new ToolStripMenuItem("Открыть папку с видео…");
            folder.Click += delegate { OpenFolder(); };
            menu.Items.Add(folder);

            var auto = new ToolStripMenuItem("Запускать вместе с Windows") { Checked = Program.GetAutostart() };
            auto.Click += delegate { Program.SetAutostart(!Program.GetAutostart(), Application.ExecutablePath); };
            menu.Items.Add(auto);

            menu.Items.Add(new ToolStripSeparator());

            var uninstall = new ToolStripMenuItem("Удалить с компьютера…");
            uninstall.Click += delegate { Uninstall(); };
            menu.Items.Add(uninstall);

            var quit = new ToolStripMenuItem("Выйти");
            quit.Click += delegate { Quit(); };
            menu.Items.Add(quit);
        }

        void OpenFolder()
        {
            Directory.CreateDirectory(Program.VideosDir);
            Process.Start("explorer.exe", "\"" + Program.VideosDir + "\"");
        }

        static string LoadLast()
        {
            try { return File.Exists(Program.SettingsFile) ? File.ReadAllText(Program.SettingsFile).Trim() : null; }
            catch { return null; }
        }

        static void SaveLast(string path)
        {
            try { File.WriteAllText(Program.SettingsFile, path); } catch { }
        }

        void Quit()
        {
            foreach (var w in windows) w.Dispose();
            windows.Clear();
            Desktop.RefreshWallpaper();     // вернуть обычные обои на место видео
            tray.Visible = false;
            tray.Dispose();
            messages.Dispose();
            ExitThread();
        }

        void Uninstall()
        {
            var r = MessageBox.Show("Удалить живые обои с компьютера?\nПапка с приложением и видео будет удалена.",
                                    "Живые обои", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (r != DialogResult.Yes) return;
            Program.SetAutostart(false, null);
            // папку удаляем через 2 секунды, когда приложение уже закроется
            var psi = new ProcessStartInfo("cmd.exe",
                "/c timeout /t 2 /nobreak >nul & rmdir /s /q \"" + Program.InstallDir + "\"")
            { CreateNoWindow = true, UseShellExecute = false };
            Process.Start(psi);
            Quit();
        }
    }

    // ------------------------------------------------------------------ окно с видео на рабочем столе
    class WmpHost : AxHost
    {
        public WmpHost() : base("6bf52a52-394a-11d3-b153-00c04f79faa6") { }   // Windows Media Player
    }

    class WallpaperWindow : Form
    {
        readonly WmpHost host;
        object player;
        int videoW = 16, videoH = 9;
        bool sized;

        public WallpaperWindow()
        {
            FormBorderStyle = FormBorderStyle.None;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.Manual;
            BackColor = Color.Black;
            host = new WmpHost();
            ((System.ComponentModel.ISupportInitialize)host).BeginInit();
            Controls.Add(host);
            ((System.ComponentModel.ISupportInitialize)host).EndInit();
        }

        protected override CreateParams CreateParams
        {
            get
            {
                var cp = base.CreateParams;
                cp.ExStyle |= 0x80 | 0x08000000;   // WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
                return cp;
            }
        }

        protected override bool ShowWithoutActivation { get { return true; } }

        public void Attach(IntPtr parent, IntPtr insertAfter, Rectangle r)
        {
            IntPtr hwnd = Handle;                       // окно создано, но ещё не показано
            Native.SetParent(hwnd, parent);
            Native.MoveWindow(hwnd, r.X, r.Y, r.Width, r.Height, true);
            if (insertAfter != IntPtr.Zero)
                Native.SetWindowPos(hwnd, insertAfter, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010);   // NOSIZE|NOMOVE|NOACTIVATE
            Show();
            player = host.GetOcx();
            Com.Set(player, "uiMode", "none");
            Com.Set(player, "stretchToFit", true);
            Com.Set(player, "enableContextMenu", false);
            object settings = Com.Get(player, "settings");
            Com.Call(settings, "setMode", "loop", true);
            Com.Set(settings, "mute", true);
            Com.Set(settings, "autoStart", true);
            DoLayout();
        }

        public void Play(string path)
        {
            if (player == null) return;
            sized = false;
            Com.Set(player, "URL", path);
            Com.Call(Com.Get(player, "controls"), "play");
        }

        public void Pause() { if (player != null) Com.Call(Com.Get(player, "controls"), "pause"); }
        public void Resume() { if (player != null) Com.Call(Com.Get(player, "controls"), "play"); }

        // Видео заполняет экран без чёрных полос: увеличиваем плеер до пропорций видео, лишнее обрезается краем окна
        public void UpdateLayoutIfReady()
        {
            if (sized || player == null) return;
            try
            {
                object media = Com.Get(player, "currentMedia");
                if (media == null) return;
                int w = Convert.ToInt32(Com.Get(media, "imageSourceWidth"));
                int h = Convert.ToInt32(Com.Get(media, "imageSourceHeight"));
                if (w > 0 && h > 0) { videoW = w; videoH = h; sized = true; DoLayout(); }
            }
            catch { }
        }

        void DoLayout()
        {
            Rectangle c = ClientRectangle;
            if (c.Width == 0 || c.Height == 0) return;
            double scale = Math.Max((double)c.Width / videoW, (double)c.Height / videoH);
            int w = (int)Math.Ceiling(videoW * scale) + 2;
            int h = (int)Math.Ceiling(videoH * scale) + 2;
            host.SetBounds((c.Width - w) / 2, (c.Height - h) / 2, w, h);
        }

        protected override void OnResize(EventArgs e) { base.OnResize(e); DoLayout(); }

        protected override void Dispose(bool disposing)
        {
            try { if (player != null) Com.Call(Com.Get(player, "controls"), "stop"); } catch { }
            player = null;
            base.Dispose(disposing);
        }
    }

    // ------------------------------------------------------------------ поиск слоя «под иконками»
    static class Desktop
    {
        // Классика (Windows 10/11): Progman создаёт отдельное окно WorkerW позади иконок — кладём видео туда.
        // Windows 11 24H2+: WorkerW теперь дочернее окно Progman — кладём видео в Progman между иконками и WorkerW.
        public static bool FindHost(out IntPtr parent, out IntPtr insertAfter)
        {
            parent = IntPtr.Zero;
            insertAfter = IntPtr.Zero;
            IntPtr progman = Native.FindWindow("Progman", null);
            if (progman == IntPtr.Zero) return false;
            IntPtr res;
            Native.SendMessageTimeout(progman, 0x052C, new IntPtr(0xD), new IntPtr(0x1), 0, 1000, out res);
            Native.SendMessageTimeout(progman, 0x052C, IntPtr.Zero, IntPtr.Zero, 0, 1000, out res);

            IntPtr workerw = IntPtr.Zero;
            Native.EnumWindows(delegate(IntPtr top, IntPtr p)
            {
                IntPtr defView = Native.FindWindowEx(top, IntPtr.Zero, "SHELLDLL_DefView", null);
                if (defView != IntPtr.Zero)
                    workerw = Native.FindWindowEx(IntPtr.Zero, top, "WorkerW", null);
                return true;
            }, IntPtr.Zero);

            if (workerw != IntPtr.Zero)
            {
                parent = workerw;
                return true;
            }

            IntPtr childDefView = Native.FindWindowEx(progman, IntPtr.Zero, "SHELLDLL_DefView", null);
            IntPtr childWorker = Native.FindWindowEx(progman, IntPtr.Zero, "WorkerW", null);
            if (childDefView != IntPtr.Zero)
            {
                parent = progman;
                insertAfter = childDefView;     // сразу под слоем иконок
                return true;
            }
            if (childWorker != IntPtr.Zero)
            {
                parent = childWorker;
                return true;
            }
            parent = progman;
            return true;
        }

        public static void RefreshWallpaper()
        {
            var sb = new StringBuilder(520);
            if (Native.SystemParametersInfo(0x0073, 520, sb, 0))      // SPI_GETDESKWALLPAPER
                Native.SystemParametersInfo(0x0014, 0, sb, 0);         // SPI_SETDESKWALLPAPER — перерисовать
        }
    }

    // Невидимое окно: ловит перезапуск проводника (тогда слой рабочего стола создаётся заново)
    class MessageWindow : NativeWindow, IDisposable
    {
        static readonly int TaskbarCreated = Native.RegisterWindowMessage("TaskbarCreated");
        public event EventHandler ExplorerRestarted;

        public MessageWindow() { CreateHandle(new CreateParams()); }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == TaskbarCreated && ExplorerRestarted != null) ExplorerRestarted(this, EventArgs.Empty);
            base.WndProc(ref m);
        }

        public void Dispose() { DestroyHandle(); }
    }

    // ------------------------------------------------------------------ COM без interop-сборок
    static class Com
    {
        public static object Get(object o, string name)
        {
            return o.GetType().InvokeMember(name, BindingFlags.GetProperty, null, o, null);
        }

        public static void Set(object o, string name, object value)
        {
            o.GetType().InvokeMember(name, BindingFlags.SetProperty, null, o, new[] { value });
        }

        public static object Call(object o, string name, params object[] args)
        {
            return o.GetType().InvokeMember(name, BindingFlags.InvokeMethod, null, o, args);
        }
    }

    static class Native
    {
        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindow(string cls, string title);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string title);
        [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc proc, IntPtr lParam);
        [DllImport("user32.dll")] public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam, uint flags, uint timeout, out IntPtr result);
        [DllImport("user32.dll")] public static extern IntPtr SetParent(IntPtr child, IntPtr parent);
        [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hWnd, int x, int y, int w, int h, bool repaint);
        [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int RegisterWindowMessage(string name);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern bool SystemParametersInfo(uint action, uint param, StringBuilder pv, uint winIni);
    }
}
