@echo off
del "C:\Projects\Sight\runs\phase_k\k7_offline\probe.txt" 2>nul
del "C:\Projects\Sight\runs\phase_k\k7_offline\probe_err.txt" 2>nul
"C:\Projects\Sight\.venv-d3rlpy\Scripts\python.exe" -c "import scipy,sklearn,d3rlpy;open(r'C:\Projects\Sight\runs\phase_k\k7_offline\probe.txt','w').write('PROBE_OK scipy='+scipy.__file__)" 2> "C:\Projects\Sight\runs\phase_k\k7_offline\probe_err.txt"
