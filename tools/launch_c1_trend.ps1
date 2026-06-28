$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\Projects\Sight\tools\run_c1_trend.bat'}
Write-Output ("PID=" + $r.ProcessId + " RET=" + $r.ReturnValue)
