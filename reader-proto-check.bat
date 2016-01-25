@echo off
setlocal
cd %~dp0
for %%f in (%*) do set a_%%f=1

@REM optionally do clean ?

set ACML_FMA=0
set CYGWIN_BIN=c:\cygwin64\bin
if not exist %CYGWIN_BIN% (
    set CYGWIN_BIN=c:\cygwin\bin
    if not exist %CYGWIN_BIN% (
        echo Can't find Cygwin, is it installed?
        exit /b 1
    )
)
echo on

@REM TODO need to get into working state again:
set a_nounittests=1
set a_nospeech=1

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
if errorlevel 1 exit /b 1

if not defined a_nodebug (
    msbuild /m /p:Platform=x64 /p:Configuration=Debug CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_notests (
if not defined a_nounittests (
        .\x64\Debug\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop1
        if errorlevel 1 exit /b 1
)
)
)

if not defined a_norelease (
    msbuild /m /p:Platform=x64 /p:Configuration=Release CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_notests (
if not defined a_nounittests (
    .\x64\Release\UnitTests\ReaderTests.exe -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop1
    if errorlevel 1 exit /b 1
)
)
)

set PATH=%PATH%;%CYGWIN_BIN%

if not defined a_nospeech (
if not defined a_noe2e (
if not defined a_notests (
if not defined a_norelease (
if not defined a_nogpu (
    python2.7.exe Tests/TestDriver.py run -d gpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/TestDriver.py run -d cpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

if not defined a_nodebug (
if not defined a_nogpu (
    python2.7.exe Tests/TestDriver.py run -d gpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/TestDriver.py run -d cpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)
)
)
)

if not defined a_notests (
if not defined a_noimage (
if not defined a_nogpu (
   
    cd ImageTest

    rmdir /s /q _out

    if not exist Data\image.nypl.org-1564068.jpg echo Get the Data first.&exit /b 1

if not defined a_norelease (
    ..\x64\Release\CNTK.exe configFile=AlexNet.config ConfigName=Release "Train=[reader=[readerType=NewImageReader]]"
    echo BASELINE
    findstr /c:"Finished Epoch" .\Release_Base.log
    echo CURRENT
    findstr /c:"Finished Epoch" .\_out\Release_Train.log
)

if not defined a_nodebug (
    ..\x64\Debug\CNTK.exe configFile=AlexNet.config ConfigName=Debug "Train=[reader=[readerType=NewImageReader]]"
    echo BASELINE
    findstr /c:"Finished Epoch" .\Debug_Base.log
    echo CURRENT
    findstr /c:"Finished Epoch" .\_out\Debug_Train.log
)

    cd ..
)
)
)
