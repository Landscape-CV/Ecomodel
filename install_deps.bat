@echo off
echo Installing core requirements...
.\.venv\Scripts\python.exe -m pip install -r reqs.txt
echo.
echo Installing fake-bpy-module...
.\.venv\Scripts\python.exe -m pip install fake-bpy-module-latest
echo.
echo Done!
