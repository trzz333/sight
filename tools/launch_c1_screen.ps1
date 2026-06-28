param([int]$Seed = 0)
# Launch the C1 ES screen detached from this shell via WMI. This console
# pattern is the one PROVEN to run (12-gen trend EXIT 0; 6-gen screen before an
# external window-CLOSE event). The forrtl error 200 abort is neutralised by
# FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 set inside run_c1_screen.bat.
#
# REJECTED (2026-06-28, evidence on disk): DETACHED_PROCESS (CreateFlags=8) to
# drop the console entirely. It silently killed the worker pool — multiprocessing
# spawn + Godot subprocs get no valid std handles with no console, so the run
# died before logging anything (runs\phase_n\c1_smoke_detach: START only, no
# gens, no sentinel, no surviving process). The correct console-less home is an
# NSSM service (valid I/O redirection + auto-restart), set up separately for the
# full 8h run; see docs\phase-n-foolproof-design.md.
$bat = 'C:\Projects\Sight\tools\run_c1_screen.bat'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=("cmd.exe /c " + $bat + " " + $Seed)}
Write-Output ("PID=" + $r.ProcessId + " RET=" + $r.ReturnValue + " SEED=" + $Seed)
