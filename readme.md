# Signal Viewer for multiple LSL streams
This is a GUI app to view and save data streaming from the LSL (LabStreamingLayer). It is designed for EEG acquisition but one can also use it to collect other physiological data, XBOX controller, or any data streamed by LSL.

## Features
* Provide a solution to compile exe with pyinstaller.
* Support plotting multiple LSL streams at once, including both data and marker (event) inlets.
* Use config.ini to handle different streams or default values w/o compiling again.
* Can save (raw) data as csv file for each stream.
* Support bandpass/bandstop filters and re-referencing on-the-fly.
* Two modes of plotting.
* Plot spectrogram of specified channel.
* Support ASR (artifact subspace removal) to process signals in real time.
* Some freedom to revise the default layout by modifying the mainwindow.ui (editable by Qt Designer).

## Known issue(s)
* First marker received will always cause the app to freeze a bit.

## How to write config.ini
* Please read the comments in the config.ini.

## To compile with pyinstaller by yourself:
For who are interested in compiling an executable file. Please try to follow the items below to compile. The compiled file should be around 101 MB on windows. This version already tries to reduce the size by avoiding unnecessary libraries in both coding and packages.\
0. (Optional) To have a smaller file, it is recommended to use the provided env_anaconda.yaml file to create an environment in Anaconda.
1. Locate the folder of the meegkit package to open testing.py and denoise.py (under utils).
2. Comment lines related to matplotlib in the above two files.
3. Locate utils/asr.py, find the function geometric_median and add "return y1" at the last line, as data must be returned even if convergence cannot be met.
4. Open pylsl.py.
5. Comment lines 1278-79 (should be like below)\
    libpath = next(find_liblsl_libraries())\
    lib = CDLL(libpath)
6. Add lib = CDLL('./lsl.dll') after the two lines above
7. Open command prompt to run pyinstaller with the command:\
    "pyinstaller --onefile main.spec" or\
    "pyinstaller --onefile main_no_console.spec" if prefer not to have a console to show logs
8. Compiled exe should appear under the dist folder
9. The exe file must locate in a folder that has config.ini AND mainwindow.ui