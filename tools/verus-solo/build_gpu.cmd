@echo off
setlocal
cd /d "%~dp0"
if not defined GPU_ARCH set GPU_ARCH=sm_89
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "soloVs=%%i"
if not defined soloVs (
  echo Visual Studio C++ tools not found
  exit /b 1
)
call "%soloVs%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
if not exist build mkdir build
nvcc -O3 -std=c++17 -arch=%GPU_ARCH% --use_fast_math -DVERUS_SOLO_STANDALONE -I..\..\algos -Xcompiler "/MT /EHsc /DNOMINMAX" -c solo_gpu.cu -o build\solo_gpu.obj || exit /b 1
cl /nologo /MT /O2 /arch:AVX2 /I..\..\algos\verus /c ..\..\algos\verus\verus_host.c ..\..\algos\verus\haraka.c ..\..\algos\verus\haraka_portable.c ..\..\algos\verus\verus_clhash.c /Fo:build\ || exit /b 1
link /nologo /out:solo_gpu.exe /libpath:"%CUDA_PATH%\lib\x64" build\solo_gpu.obj build\verus_host.obj build\haraka.obj build\haraka_portable.obj build\verus_clhash.obj cudart_static.lib kernel32.lib user32.lib advapi32.lib shell32.lib ole32.lib || exit /b 1
