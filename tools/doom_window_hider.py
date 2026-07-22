"""Standalone daemon: neutralize ViZDoom engine windows before they paint.

Why: the engine (vizdoom.exe, ZDoom 2.8.1) creates its main window VISIBLE
during init, ~0.8s before the viz_window_hidden handshake hides it, and it
re-shows/re-centers the window repeatedly during startup, so any single
hide/move loses the race. Prevention also fails (verified 2026-07-21/22:
SDL dummy driver, alternate desktop via lpDesktop - the spawn lands on the
Default desktop, ZDoom -nostartup, ini win_x/win_y parking - the engine
recenters, per-worker in-process hooks - GIL contention delays the callback
100-300ms, and a single GLOBAL WinEvent hook - desktop-wide OBJECT_CREATE
volume floods the queue with seconds of lag).

Working design: poll the process table every 50ms for new vizdoom.exe
pids (the engine loads WADs for ~0.5s before creating its window), then
install a PID-SCOPED WinEvent hook per engine. Event volume per hook is a
handful, delivery is single-digit ms. On each window event: give the
window an EMPTY region (paints zero pixels regardless of visibility),
strip its taskbar button (WS_EX_TOOLWINDOW), move it off-screen, hide it.
The engine can re-show all it wants; a region-less toolwindow off-screen
is imperceptible from the first event on.

Runs forever under pythonw (no console). Single-instance via named mutex.
Startup entry: sight-doom-hider.vbs. Log: runs\\vzd\\doom_hider.log.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import time

LOG = r"C:\Projects\Sight\runs\vzd\doom_hider.log"
MUTEX = "Global\\SightDoomWindowHider"
ERROR_ALREADY_EXISTS = 183
SWP = 0x0010 | 0x0004 | 0x0001  # NOACTIVATE | NOZORDER | NOSIZE

u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32
g32 = ctypes.windll.gdi32
# Separate handle for the mutex check: plain windll's GetLastError() is
# clobbered by ctypes' own intervening Win32 calls (observed 2026-07-22:
# two instances both read 0 and coexisted for 12h). use_last_error=True
# snapshots the error code at FFI return; read it with get_last_error().
_k32e = ctypes.WinDLL("kernel32", use_last_error=True)


def log(msg: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def park(hwnd: int, why: str = "") -> None:
    # skip if already parked (empty region): keeps logs and calls quiet
    probe = g32.CreateRectRgn(0, 0, 0, 0)
    kind = u32.GetWindowRgn(hwnd, probe)
    g32.DeleteObject(probe)
    if kind == 1:  # NULLREGION already applied
        return
    rgn = g32.CreateRectRgn(0, 0, 0, 0)
    ok = u32.SetWindowRgn(hwnd, rgn, True)  # system owns rgn on success
    GWL_EXSTYLE, WS_EX_TOOLWINDOW = -20, 0x00000080
    ex = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    u32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_TOOLWINDOW)
    u32.SetWindowPos(hwnd, None, -32000, -32000, 0, 0, SWP)
    u32.ShowWindow(hwnd, 0)  # SW_HIDE
    log(f"park hwnd={hwnd} via={why} rgn_ok={ok}")


def doom_pids() -> set[int]:
    TH32CS_SNAPPROCESS = 2

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD),
            ("th32ProcessID", wt.DWORD), ("th32DefaultHeapID",
                                          ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", wt.LONG),
            ("dwFlags", wt.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    pids: set[int] = set()
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == -1:
        return pids
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if k32.Process32FirstW(snap, ctypes.byref(pe)):
            while True:
                if pe.szExeFile.lower() == "vizdoom.exe":
                    pids.add(pe.th32ProcessID)
                if not k32.Process32NextW(snap, ctypes.byref(pe)):
                    break
    finally:
        k32.CloseHandle(snap)
    return pids


def sweep_pids(pids: set[int], why: str) -> None:
    """Park windows owned by any of the given pids."""
    if not pids:
        return
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def cb(hwnd, lp):
        owner = wt.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in pids:
            park(hwnd, why)
        return True

    u32.EnumWindows(WNDENUMPROC(cb), 0)


def main() -> None:
    _k32e.CreateMutexW(None, False, MUTEX)
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        # Duplicate instance: nothing to clean up, and normal sys.exit has
        # been observed to wedge in interpreter shutdown under pythonw
        # (pid 7888, 2026-07-22, single thread parked forever). Hard exit.
        import os
        os._exit(0)

    WINEVENTPROC = ctypes.WINFUNCTYPE(
        None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG,
        wt.DWORD, wt.DWORD)

    def cb(hook, event, hwnd, obj_id, child_id, thread_id, ms):
        if obj_id == 0 and hwnd:  # OBJID_WINDOW; hook is already pid-scoped
            park(hwnd, f"hook:{hex(event)}")

    cbref = WINEVENTPROC(cb)  # must outlive all hooks
    EVENT_OBJECT_CREATE, EVENT_OBJECT_SHOW = 0x8000, 0x8002
    hooks: dict[int, int] = {}  # pid -> hook handle
    log("daemon up (pid-scoped hooks + 50ms sweep backstop)")

    PM_REMOVE = 1
    msg = wt.MSG()
    while True:
        live = doom_pids()
        for pid in live - hooks.keys():
            h = u32.SetWinEventHook(EVENT_OBJECT_CREATE, EVENT_OBJECT_SHOW,
                                    None, cbref, pid, 0, 0)
            hooks[pid] = h
            log(f"hooked pid={pid} handle={h}")
        for pid in list(hooks.keys() - live):
            u32.UnhookWinEvent(hooks.pop(pid))
        # Backstop: hook delivery has proven unreliable for fast engine
        # respawns, so sweep every cycle. Bounds any flash at ~50ms even
        # if no hook event ever arrives.
        sweep_pids(live, "sweep")
        # Wake IMMEDIATELY when a hook event arrives, or after 50ms.
        QS_ALLINPUT = 0x04FF
        u32.MsgWaitForMultipleObjects(0, None, False, 50, QS_ALLINPUT)
        while u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            u32.TranslateMessage(ctypes.byref(msg))
            u32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # owner died silently 2026-07-22 ~02:1x; want evidence
        import traceback
        log("daemon crash:\n" + traceback.format_exc())
        raise
