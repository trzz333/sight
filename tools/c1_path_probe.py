import os
exe = (r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
       r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
       r"\Godot_v4.6.2-stable_win64.exe")
proj = r"C:\Projects\Sight\games\signal-dodge"
print("EXE exists:", os.path.isfile(exe))
print("PROJ dir exists:", os.path.isdir(proj))
print("project.godot exists:", os.path.isfile(os.path.join(proj, "project.godot")))
