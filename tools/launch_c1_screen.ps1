param([int]$Seed = 0)
$bat = 'C:\Projects\Sight\tools\run_c1_screen.bat'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=("cmd.exe /c " + $bat + " " + $Seed)}
Write-Output ("PID=" + $r.ProcessId + " RET=" + $r.ReturnValue + " SEED=" + $Seed)
