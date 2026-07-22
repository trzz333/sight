# Persistent popup evidence collector. Logs every process that acquires a
# visible top-level window, forever, with rotation. When Jeff sees a popup,
# the offender's pid/path/title/time is already in this log.
param([string]$Out = "C:\Projects\Sight\runs\vzd\window_watch.log")
Add-Type -Namespace SightWin -Name Native -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
public struct RECT { public int Left, Top, Right, Bottom; }
'@
function Get-RectTag([IntPtr]$h) {
    $r = New-Object SightWin.Native+RECT
    if ([SightWin.Native]::GetWindowRect($h, [ref]$r)) {
        return "rect=($($r.Left),$($r.Top))-($($r.Right),$($r.Bottom))"
    }
    return "rect=?"
}
$seen = @{}
Get-Process | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object {
    $seen["$($_.Id)"] = $true }
"[watch start $(Get-Date -Format o)] baseline $($seen.Count) windowed procs" |
    Out-File -FilePath $Out -Append -Encoding utf8
$i = 0
while ($true) {
    Get-Process | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object {
        if (-not $seen.ContainsKey("$($_.Id)")) {
            $seen["$($_.Id)"] = $true
            "[{0}] NEW WINDOW pid={1} proc={2} {3} title='{4}' path='{5}'" -f `
                (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'), $_.Id,
                $_.ProcessName, (Get-RectTag $_.MainWindowHandle),
                $_.MainWindowTitle, $_.Path |
                Out-File -FilePath $Out -Append -Encoding utf8
        }
    }
    $i++
    if ($i % 2400 -eq 0) {  # ~20 min: prune dead pids, rotate if large
        $alive = @{}; Get-Process | ForEach-Object { $alive["$($_.Id)"] = 1 }
        $pruned = @{}
        foreach ($k in $seen.Keys) {
            if ($alive.ContainsKey($k)) { $pruned[$k] = $true } }
        $seen = $pruned
        if ((Get-Item $Out -ErrorAction SilentlyContinue).Length -gt 2MB) {
            Get-Content $Out -Tail 2000 | Set-Content $Out -Encoding utf8 }
    }
    Start-Sleep -Milliseconds 500
}
