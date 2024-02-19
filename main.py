"""
@author: Ping-Keng Jao
"""
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout)
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore, QtGui
import sys
from os.path import dirname

# Note: this file includes several commented lines using pandas. However, use of pandas
# was discarded because pyinstaller will lead to a large exe file due to used libraries.
# import pandas as pd
import numpy as np
import csv
import pylsl

from typing import List
from math import ceil

from signal_proc_pipeline import reref, create_iir_filter, create_fir_filter, filtClass
from meegkit import asr

from LSL_inlets import Inlet, DataInlet, MarkerInlet, dataBuffer2D, dataBuffer1D

## add import below because pyinstaller cannot find them properly even with a spec file...
# import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5
# import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5
# import pyqtgraph.imageview.ImageViewTemplate_pyqt5
# import sklearn.utils._cython_blas
# import sklearn.utils._weight_vector
from sklearn.metrics import pairwise_distances
import sklearn.metrics._pairwise_distances_reduction._middle_term_computer

import configparser
def read_config():
    cfg = configparser.ConfigParser()
    cfg.read('./config.ini')
    global update_interval, pull_interval, fudge_factor, labelStyles, ch_colors, target_streams, stream_configs, channel_per_column
    update_interval = int(cfg.getfloat('Time', 'update_interval'))
    pull_interval = int(cfg.getfloat('Time', 'pull_interval'))
    fudge_factor = cfg.getfloat('Time', 'fudge_factor')
    fudge_factor = fudge_factor * pull_interval
    labelStyles = {'colors': None, 'font-size': None}
    labelStyles['color'] = cfg.get('Figure', 'label_color')
    labelStyles['font-size'] = cfg.get('Figure', 'fontsize')
    target_streams = parseStrList(cfg.get('StreamNames', 'Names'))
    stream_configs = {x: {} for x in target_streams}
    for ts in target_streams:
        stream_configs[ts]['spacing'] = cfg.getfloat('Stream_spacing', ts)
        stream_configs[ts]['add_event'] = cfg.getboolean('Stream_add_event_stream', ts)
        stream_configs[ts]['channel_names'] = parseStrList(cfg.get('Stream_Channels', ts))
        stream_configs[ts]['channel_count'] = len(stream_configs[ts]['channel_names'])
        stream_configs[ts]['separate_channels'] = cfg.getboolean('Stream_separate_channels', ts)
        yRange = parseStrList(cfg.get('Stream_yrange', ts))
        stream_configs[ts]['yRange'] = [float(y) for y in yRange]
        init_bandpass = parseStrList(cfg.get('Stream_init_bandpass', ts))
        init_bandstop = parseStrList(cfg.get('Stream_init_bandstop', ts))
        stream_configs[ts]['init_bandpass'] = [float(y) for y in init_bandpass]
        stream_configs[ts]['init_bandstop'] = [float(y) for y in init_bandstop]
        stream_configs[ts]['add_legend'] = cfg.getboolean('Stream_add_legend', ts)
        stream_configs[ts]['stretch_factor'] = cfg.getint('Stream_stretch_factor', ts)
        if stream_configs[ts]['stretch_factor'] < 0:
            stream_configs[ts]['stretch_factor'] = stream_configs[ts]['channel_count']
    ch_colors = parseStrList(cfg.get('Figure', 'ch_colors'))
    channel_per_column = cfg.getint('Figure', 'channel_per_column')

def parseStrList(x):
    y = x.split(',')
    for i, s in enumerate(y):
        y[i] = s.strip()
    return y

def read_asr_config():
    cfg = configparser.ConfigParser()
    cfg.read('./config.ini')
    bandpass_low = cfg.getfloat('ASR', 'bandpass_low')
    bandpass_high = cfg.getfloat('ASR', 'bandpass_high')
    cutoff = cfg.getfloat('ASR', 'cutoff')
    blocksize = cfg.getint('ASR', 'blocksize')
    win_len = cfg.getfloat('ASR', 'win_len')
    win_overlap = cfg.getfloat('ASR', 'win_overlap')
    max_dropout_fraction = cfg.getfloat('ASR', 'max_dropout_fraction')
    min_clean_fraction = cfg.getfloat('ASR', 'min_clean_fraction')
    method = cfg.get('ASR', 'method')
    estimator = cfg.get('ASR', 'estimator')
    return bandpass_low, bandpass_high, cutoff, blocksize, win_len, win_overlap, max_dropout_fraction, min_clean_fraction, method, estimator

def read_ui_config():
    cfg = configparser.ConfigParser()
    cfg.read('./config.ini')
    return cfg.get('UI', 'fontsize')

class SpectrogramWidget(PlotWidget):
    def __init__(self, ch_idx, win_size, step_size, srate, plot_duration, plot_mode, streamname, stream_config, windowflag=None):
        super(SpectrogramWidget, self).__init__()
        self.buff = dataBuffer1D(ceil(plot_duration*srate*2))
        self.ch_idx = ch_idx
        self.win_size = int(win_size)
        self.step_size = int(step_size)
        self.n_window = int((plot_duration*srate)//step_size)
        self.srate = srate
        self.streamname = streamname
        self.stream_config = stream_config
        self.imv = pg.ImageView()
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self.img_array = np.full((self.n_window, int(self.win_size/2+1)), fill_value=np.nan)
        self.plot_mode = plot_mode
        self.ptr = 0 # for refresh plotting mode

        self.setWindowTitle(self.stream_config['channel_names'][ch_idx])
        self.setTitle(self.stream_config['channel_names'][ch_idx])
        if windowflag is not None: self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-10,40])

        freq = np.arange((self.win_size/2)+1)/(float(self.win_size)/self.srate)

        self.xaxis = self.getAxis('bottom')
        self.yaxis = self.getAxis('left')
        self.setXRange(0, self.n_window, padding=0)
        self.setYRange(0, len(freq), padding=0)
        # handle xtcik below
        xtick = np.linspace(0, self.n_window, plot_duration+1)
        self.xTickLabels = [str(i) for i in range(plot_duration+1)]
        self.xaxis.setTicks([[(x, y) for x, y in zip(xtick, self.xTickLabels)]])
        # handle ytick below
        target_res_spacing = 10
        target_freq = [i for i in range(0, int(freq[-1]), target_res_spacing)]
        if target_freq[-1] != freq[-1]: target_freq.append(freq[-1])
        target_freq = np.array(target_freq)
        target_freq = np.repeat(target_freq[np.newaxis], len(freq), axis=0)
        distance = np.abs(np.subtract(freq[:, np.newaxis], target_freq))
        ytick = np.argmin(distance, axis=0)
        self.yTickLabels = [str(freq[i]) for i in ytick]
        self.yaxis.setTicks([[(x, y) for x, y in zip(ytick, self.yTickLabels)]])

        self.bar = pg.ColorBarItem(values= (-20, 40), colorMap=cmap) # prepare interactive color bar

        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', 'Time', units='s')

        self.win = np.hanning(self.win_size)
        self.show()

    def update(self, chunk):
        self.buff.addData(chunk)
        data, size = self.buff.fetchData(self.win_size, self.step_size)
        while size > 0: # squeeze out all plottable data
            # normalized, windowed frequencies in data chunk
            spec = np.fft.rfft(data*self.win) / self.win_size
            # get magnitude
            psd = abs(spec)
            # convert to dB scale
            psd = 20 * np.log10(psd)

            if self.plot_mode == 'Scroll':
                # roll down one and replace leading edge with new data
                self.img_array = np.roll(self.img_array, -1, 0)
                self.img_array[-1:] = psd
            elif self.plot_mode == 'Refresh':
                self.img_array[self.ptr, :] = psd
                self.ptr = (self.ptr+1)%self.n_window
            else:
                print(f'This should not happen, what is mode {self.plot_mode}?')
            data, size = self.buff.fetchData(self.win_size, self.step_size)

        self.img.setImage(self.img_array, autoLevels=False)
        # Have ColorBarItem control colors of img and appear in 'plot':
        self.bar.setImageItem(self.img, insert_in=self.getPlotItem())


class SignalPlotWidget(PlotWidget):
    def __init__(self, parent, inlets, plot_duration, spacing, plot_mode, streamname, stream_config):
        super(SignalPlotWidget, self).__init__(parent=parent)
        self.srate = 250 # temp value, should be override
        self.inlets = inlets
        self.plot_duration = plot_duration
        self.pause = False
        self.save = False
        self.spacing = spacing
        self.plot_mode = plot_mode
        self.saveName = f'{streamname}.csv'
        self.time_startsaving = pylsl.local_clock()
        self.saveEvent = False
        self.streamname = streamname
        self.setEventFile()
        self.stream_config = stream_config
        self.init_plot()
        self.reref_idx = 0
        self.proc_fun = [lambda x: reref(x, idx=self.reref_idx)] # allocate for reref, bandpass, bandstop, ASR
        self.proc_fun_name = ['reref']
        self.buff = [] # for the spectrogram plot
        self.spectrogram: List[SpectrogramWidget] = []
        self.prev_mintime = pylsl.local_clock()

        # create a timer that will move the view every update_interval ms, where the timer only start after connecting to LSL
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.scroll)

        # create a timer that will pull and add new data occasionally, where the timer only start after connecting to LSL
        self.pull_timer = QtCore.QTimer()
        self.pull_timer.timeout.connect(self.update)

    def closeEvent(self, a0: QtGui.QCloseEvent):
        for s in self.spectrogram:
            s.close(a0)

    def changeReref(self, idx):
        self.reref_idx = idx
        self.proc_fun[0] = lambda x: reref(x, self.reref_idx)

    def reset_bandpass_filter(self):
        exist, index = self.search_proc_fun('bandpass')
        if exist:
            self.proc_fun[index] = lambda x: x # do nothing

    def reset_bandstop_filter(self):
        exist, index = self.search_proc_fun('bandstop')
        if exist:
            self.proc_fun[index] = lambda x: x # do nothing

    def calibrate_asr(self):
        folder = dirname(self.saveName)
        calibrate_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', folder ,"CSV (*.csv)")
        if not hasattr(self, 'srate'):
            self.srate = 250
        bandpass_low, bandpass_high, cutoff, blocksize, win_len, win_overlap, max_dropout_fraction, min_clean_fraction, method, estimator = read_asr_config()
        self.ASR = asr.ASR(sfreq=self.srate, cutoff=cutoff, blocksize=blocksize, win_len=win_len, win_overlap=win_overlap, max_dropout_fraction=max_dropout_fraction, \
            min_clean_fraction=min_clean_fraction, method=method, estimator=estimator)
        # read file
        # data = pd.read_csv(calibrate_file)
        # data = data[stream_configs[self.streamname]['channel_names']]
        data = np.genfromtxt(calibrate_file, dtype=float, delimiter=',', skip_header=True)
        data = data[:, 1:] # assuming the 0th column is always time stamp
        # run pre-processing
        bandpass_exist, index = self.search_proc_fun('bandpass')
        if not bandpass_exist:
            self.init_bandpass_filter(low=bandpass_low, high=bandpass_high, name=self.streamname)
        
        ASR_exist, index = self.search_proc_fun('ASR')
        if ASR_exist:
            self.proc_fun.pop(index)
            self.proc_fun_name.pop(index)

        for f in self.proc_fun:
            data = f(data)
        
        # train for ASR
        self.ASR.fit(data.transpose())

        self.proc_fun.append(lambda x: self.ASR.transform(x.transpose()).transpose())
        self.proc_fun_name.append('ASR')

    def switch_asr(self):
        ASR_exist, index = self.search_proc_fun('ASR')
        if ASR_exist:
            self.proc_fun.pop(index)
            self.proc_fun_name.pop(index)
            return False
        else:
            self.proc_fun.append(lambda x: self.ASR.transform(x.transpose()).transpose())
            self.proc_fun_name.append('ASR')
            return True

    def search_proc_fun(self, target_name):
        if target_name in self.proc_fun_name:
            index = self.proc_fun_name.index(target_name)
            return True, index
        else:
            return False, -1

    def init_bandpass_filter(self, low, high, name, verbose=True):
        print(f"Applying a bandpass filter ({low}, {high}) Hz on {self.streamname} with {self.stream_config['channel_count']} channel(s) and sampling rate = {self.srate} Hz")
        assert(high > low)
        # FIR
        # b = create_fir_filter(fs=self.srate, cutoff=[low, high], nTap='8s', filter_type='bandpass')
        # b = create_fir_filter(fs=self.srate, cutoff=[low, high], nTap=np.int32(self.srate*1), filter_type='bandpass')
        # a = 1
        # IIR
        b, a = create_iir_filter(fs=self.srate, cutoff=[low, high], order=4, filter_type='bandpass', ftype='butter', title=name, verbose=verbose)
        # b, a = create_iir_filter(fs=self.srate, cutoff=[low, high], order=2, filter_type='bandpass', ftype='butter')
        self.filter = filtClass(b, a, self.stream_config['channel_count'])

        exist, index = self.search_proc_fun('bandpass')
        if not exist:
            self.proc_fun.insert(1, lambda x: self.filter.filter(x))
            self.proc_fun_name.insert(1, 'bandpass')
        else:
            self.proc_fun[index] = lambda x: self.filter.filter(x)

    def init_bandstop_filter(self, low, high, name, verbose=True):
        exist, index = self.search_proc_fun('bandstop')
        if exist:
            self.reset_bandstop_filter()
            print(f'Disabling bandstop filter as requested. Press bandstop again to activate')
            self.proc_fun.pop(index)
            self.proc_fun_name.pop(index)
            return False
        else:
            print(f"Applying a bandstop filter ({low}, {high}) Hz on {self.streamname} with {self.stream_config['channel_count']} channel(s) and sampling rate = {self.srate} Hz")
            assert(high > low)
            # FIR
            # b = create_fir_filter(fs=self.srate, cutoff=[low, high], nTap='8s', filter_type='bandpass')
            # b = create_fir_filter(fs=self.srate, cutoff=[low, high], nTap=np.int32(self.srate*1), filter_type='bandpass')
            # a = 1
            # IIR
            b, a = create_iir_filter(fs=self.srate, cutoff=[low, high], order=4, filter_type='bandstop', ftype='butter', title=name, verbose=verbose)
            # b, a = create_iir_filter(fs=self.srate, cutoff=[low, high], order=2, filter_type='bandpass', ftype='butter')
            self.filter_stop = filtClass(b, a, self.stream_config['channel_count'])            
            self.proc_fun.insert(2, lambda x: self.filter_stop.filter(x))
            self.proc_fun_name.insert(2, 'bandstop')
            return True

    def addSpectrogram(self, idx, win_size, step_size, windowFlag=None):
        self.spectrogram.append(SpectrogramWidget(idx, win_size, step_size, self.srate, self.plot_duration, self.plot_mode, self.streamname, stream_configs[self.streamname], windowFlag))

    def changePauseVariable(self):
        self.pause = not self.pause

    def clear_lines(self):
        for inlet in self.inlets:
            if isinstance(inlet, MarkerInlet):
                inlet.clear_all_markers()
            else:
                for c in inlet.curves:
                    c.clear()

    def removeClosedWidget(self):
        for i in range(len(self.spectrogram)-1, -1, -1):
            if not self.spectrogram[i].isVisible(): self.spectrogram.pop(i) # not visible = user closed

    def scroll(self):
        """Move the view so the data appears to scroll"""
        plot_time = pylsl.local_clock()
        self.removeClosedWidget()
        # make xticks fixed
        if not self.pause:
            if self.plot_mode == 'Scroll':
                ff = fudge_factor/1000 # convert from ms to s
                pi = pull_interval/1000
                self.plt.setXRange(plot_time - self.plot_duration + ff - pi, plot_time - ff - pi, padding=0)
                xtick = np.linspace(plot_time - self.plot_duration + ff- pi, plot_time + ff - pi, self.plot_duration+1)
            elif self.plot_mode == 'Refresh':
                self.plt.setXRange(0, self.plot_duration*self.srate, padding=0)
                xtick = np.linspace(0, self.srate*self.plot_duration, self.plot_duration+1)
            else:
                print(f'This should not happen, what is mode {self.plot_mode}?')
            self.xaxis.setTicks([[(x, y) for x, y in zip(xtick, self.xTickLabels)]])

    def update(self):
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        curtime = pylsl.local_clock()
        mintime = curtime - self.plot_duration - pull_interval # - pull_interval to make not to disappear so quick
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        tmp_data = None
        prev_x = 0
        ts = None
        events = {'time': [], 'event': []}
        if not self.pause:
            for inlet in self.inlets:
                if isinstance(inlet, MarkerInlet):
                    continue
                else: # signals
                    procssed_data = inlet.pull_and_plot(mintime, self.yshift, self.proc_fun, self.plot_mode)
                    prev_x = inlet.prev_x
                    x_left_refresh = inlet.ref_time_at_left
                    if inlet.chunk_size > 0:
                        y = inlet.buffer[0:inlet.chunk_size, :]
                        ts = np.array(inlet.timestamps) - self.time_startsaving
                        for spc in self.spectrogram:
                            spc.update(y[:, spc.ch_idx])
                        if tmp_data is None:
                            tmp_data = np.concatenate((ts[:, None], y), axis=1)
                        else:
                            tmp_data = np.concatenate((tmp_data, y), axis=1)

            for inlet in self.inlets: # plot event later than signal due to prev_x
                if isinstance(inlet, MarkerInlet):
                    inlet.pull_and_plot(plot_time=mintime, time_left_refresh=x_left_refresh, idx_prev_x_signal=prev_x, plot_mode=self.plot_mode, srate=self.srate, prev_x=prev_x, sname=self.streamname)
                    events['event'] += inlet.new_events['event']
                    events['time'] += [x - self.time_startsaving for x in inlet.new_events['time']]
            # save event data, not aligned to signals due to irregular sampling rate
            if self.save and len(events['time']) != 0 and self.saveEvent:
                # print(events)
                # df = pd.DataFrame(events)
                # df.to_csv(self.saveFile_event, mode='a',  =False, index=False, na_rep='NULL')
                # tmp_data = np.hstack((np.array(events['time'], events['event'])))

                # tmp_data = [np.array(events['time'] + events['event'][0]) for _ in events['event']]
                # print(f'tmp_data: {tmp_data}')
                # self.eventwriter.writerows(tmp_data)
                list_eventDict = [{'time': events['time'][i], 'event': events['event'][i]} for i in range(len(events['time']))]
                self.eventwriter.writerows(list_eventDict)
                # print(events['time'], ts)
            # save data for signals
            if self.save and tmp_data is not None:
                ## tmp_data = np.concatenate((tmp_data, np.full((tmp_data.shape[0], 1), event)), axis=1)
                # df = pd.DataFrame(tmp_data.tolist())
                # df.to_csv(self.saveFile, mode='a', header=False, index=False, na_rep='NULL')
                # pandas above was discarded because pyinstaller will lead to an extra large exe file
                
                # choose one of the lines below, saving raw data is generally suggested.
                self.csvwriter.writerows(tmp_data) # raw data
                # self.csvwriter.writerows(procssed_data) # re-referenced, filtered, or ASRed data
        self.prev_mintime = mintime

    def startSaving(self):
        self.save = True
        # self.saveFile = open(self.saveName, mode='w', newline='') # for pandas
        # df = pd.DataFrame(columns=['Timestamp'] + self.stream_config['channel_names'])
        # df.to_csv(self.saveFile, mode='w', header=True, index=False, na_rep='NULL')
        # pandas above was discarded because pyinstaller will lead to an extra large exe file
        self.saveFile = open(self.saveName, mode='w', newline='')
        self.csvwriter = csv.writer(self.saveFile, delimiter=',', quoting=csv.QUOTE_NONE)
        self.csvwriter.writerow(['Timestamp'] + self.stream_config['channel_names'])

        if self.saveEvent:
            # self.saveFile_event = open(self.saveName_event, mode='w', newline='')
            # df = pd.DataFrame(columns=['Timestamp', 'Event'])
            # df.to_csv(self.saveFile_event, mode='w', header=True, index=False, na_rep='NULL')
            # pandas above was discarded because pyinstaller will lead to an extra large exe file
            self.saveFile_event = open(self.saveName_event, mode='w', newline='')
            self.eventwriter = csv.DictWriter(self.saveFile_event, delimiter=',', quoting=csv.QUOTE_ALL, fieldnames=['time', 'event'])
            self.eventwriter.writeheader()

    def stopSaving(self):
        self.save = False
        self.saveFile.close()
        if self.saveEvent:
            self.saveFile_event.close()

    def setEventFile(self, prefix=''):
        self.saveName_event = prefix + '_event.csv'

    def init_plot(self):
        font=QtGui.QFont()
        font.setPixelSize(20)
        self.plt = self.getPlotItem()
        self.plt.setLabel("left", "Channels", **labelStyles)
        self.plt.setLabel("bottom", "Time (s)", **labelStyles)
        self.plt.showGrid(x=True, y=True, alpha=0.1)
        self.plt.enableAutoRange(x=False, y=False)
        self.yaxis = self.plt.getAxis('left')
        self.xaxis = self.plt.getAxis('bottom')
        self.xaxis.setStyle(tickFont = font)
        self.yaxis.setStyle(tickFont = font)
        self.xTickLabels = [str(i) for i in range(self.plot_duration)]

        nch = self.stream_config['channel_count']
        if self.stream_config['separate_channels'] and nch != 1:
            single_spacing = self.spacing
            self.yshift = [single_spacing*(i-nch//2) for i in range(nch)]
            self.yaxis.setTicks([[(x, y) for x, y in zip(self.yshift, self.stream_config['channel_names'])]])
            small_margin = 0.75
            self.plt.setYRange(min(self.yshift) - single_spacing*small_margin, max(self.yshift) + single_spacing*small_margin, padding=0)
        else:
            self.yshift = [0 for _ in range(nch)]
            self.plt.setLabel("left", "", **labelStyles)
            self.plt.setYRange(self.stream_config['yRange'][0], self.stream_config['yRange'][1])

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, screen, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        #Load the UI Page and initialize UI elements and connections
        loadUi('./mainwindow.ui', self)
        read_config()
        self.graphWidgets = {}
        self.screen = screen
        self.changeStreamName()
        self.plot_duration = int(self.lineEdit_duration.text())

        self.inlets: List[Inlet] = []
        self.n_inlet = 0
        self.widget_signal = {}
        self.init_buttons()
        self.streamname_with_event = [] # record which streamname has event stream attached

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        for w in self.widget_signal.values():
            w.closeEvent(a0)
            for s in w.inlets:
                if hasattr(s, 'closeEvent'):
                    s.close_stream()
        return super().closeEvent(a0)

    def resizeEvent(self, event):
        plot_stretch, ctrl_stretch = self.getStrecchFactors()
        if hasattr(self.layout, 'setStretch'):
            self.layout.setStretch(0, plot_stretch)
            self.layout.setStretch(1, ctrl_stretch)

    def connect_to_LSL(self):
        read_config()
        self.global_clock_ref = pylsl.local_clock()
        self.comboBox_streamName.clear()
        for w in self.widget_signal.values():
            w.update_timer.stop()
            w.pull_timer.stop()
            w.reset_bandpass_filter()
            w.clear_lines()
            w.closeEvent(None)
            w.close()
        if self.widget_signal is not None:
            del self.widget_signal
            self.widget_signal = {}

        self.inlets: List[Inlet] = [] # reset the list just in case
        # firstly resolve all streams that could be shown
        self.label_status.setText("looking for streams")
        print("looking for streams")
        streams = pylsl.resolve_streams()
        # iterate over found streams, creating specialized inlet objects that will
        # handle plotting the data
        self.n_inlet = 0
        for info in streams:
            if info.type() == 'Markers':
                if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                        or info.channel_format() == pylsl.cf_undefined:
                    print('Invalid marker stream ' + info.name())
                    print('srate: '+ str(info.nominal_srate()))
                    # print(pylsl.IRREGULAR_RATE)
                    print('channel_format: '+ str(info.channel_format()))
                    # print(pylsl.cf_undefined)
                print('Adding marker inlet: ' + info.name())
                self.inlets.append(info)
                self.n_inlet += 1
            elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                    and info.channel_format() != pylsl.cf_string:
                if info.name() not in target_streams:
                    print('Found but skipped data inlet: ' + info.name() + ', which is not specified in the target_streams and config is set as not to accept extra stream')
                    continue # only handle with designed streams
                print(f"Found data inlet: {info.name()} with {info.channel_count()} channel(s) and sampling rate = {info.nominal_srate()}")
                print('Adding data inlet: ' + info.name())

                self.addNewPlotWidget(info)
                self.comboBox_streamName.addItems([info.name()])
            else:
                print('Don\'t know what to do with stream: ' + info.name() + ' whose type is: ' + info.type())
        self.label_status.setText(f"got {self.n_inlet} available stream(s)")
        # start_time = pylsl.local_clock()
        if len(self.inlets) != 0: # append event inlets to other widgets
            for w in self.widget_signal.values():
                if not stream_configs[w.streamname]['add_event']:
                    continue
                self.streamname_with_event.append(w.streamname)
                for s in self.inlets:
                    remove_limit_range = w.inlets[0].srate*1
                    print(remove_limit_range)
                    w.inlets.append(MarkerInlet(s, w.plt, self.plot_duration, self.global_clock_ref, remove_limit_range=remove_limit_range))
            self.widget_signal[self.streamname_with_event[0]].saveEvent = True

        for w in self.widget_signal.values():
            low_cut, high_cut = stream_configs[w.streamname]['init_bandpass']
            if low_cut != high_cut:
                w.init_bandpass_filter(low_cut, high_cut, name=w.streamname, verbose=False)
            low_cut, high_cut = stream_configs[w.streamname]['init_bandstop']
            if low_cut != high_cut:
                w.init_bandstop_filter(low_cut, high_cut, name=w.streamname, verbose=False)            
        self.organizeSignalWidget()

    def getfile(self):
        folder = dirname(self.lineEdit_savename.text())
        firstSaveName, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Open file', folder ,"CSV (*.csv)")
        prefix = firstSaveName[:-4]
        for i, w in enumerate(self.widget_signal.values()):
            w.saveName = prefix + f'_{w.streamname}.csv'
            w.setEventFile(prefix) # set for all, just in case

        self.lineEdit_savename.setEnabled(True)
        self.lineEdit_savename.setText(firstSaveName)
        self.lineEdit_savename.setEnabled(False)

    def saveData(self):
        time_startsaving = pylsl.local_clock()
        for w in self.widget_signal.values():
            w.startSaving()
            w.time_startsaving = time_startsaving
            print('Saving data to: ', w.saveName)
        if len(self.streamname_with_event) > 0:
            print('Saving data to: ', self.widget_signal[self.streamname_with_event[0]].saveName_event)
        kw = next(iter(self.widget_signal)) # get the first key (if ordered)
        self.label_status.setText('Saving data to: '+ self.widget_signal[kw].saveName)

    def stopSave(self):
        for w in self.widget_signal.values():
            w.stopSaving()
        print('Saving stopped')
        self.label_status.setText('Saving stopped')

    def changePlotDuration(self):
        new_dur = int(self.lineEdit_duration.text())
        if self.plot_duration != new_dur:
            self.plot_duration = new_dur
            if self.n_inlet != 0:
                self.connect_to_LSL() # since buffer sizes need to be changed

    def changeSpacing(self):
        new_spacing = float(self.lineEdit_spacing.text())
        if self.widget_signal[self.streamname].spacing != new_spacing:
            self.widget_signal[self.streamname].spacing = new_spacing
        print('Now spacing is: ', self.widget_signal[self.streamname].spacing)
        self.widget_signal[self.streamname].init_plot()

    def changePlotMode(self, mode):
        for w in self.widget_signal.values():
            w.plot_mode = mode
            w.clear_lines()
            for s in w.spectrogram:
                s.plot_mode = mode
                s.img_array = np.full_like(s.img_array, fill_value=np.nan)
        if self.n_inlet != 0:
            self.connect_to_LSL() # since x (time) changed
        for w in self.widget_signal.values():
            w.init_plot()

    def changeStreamName(self, name=None):
        self.streamname = name
        if name is not None and len(name)!=0:
            try: self.comboBox_reref.currentIndexChanged.disconnect()
            except Exception: pass # nothing connected will throw an exception
            self.comboBox_reref.clear()
            self.comboBox_reref.addItems(['None'] + stream_configs[name]['channel_names'] + ['CAR'])
            self.comboBox_reref.setCurrentIndex(self.widget_signal[name].reref_idx)
            self.comboBox_reref.currentIndexChanged.connect(self.widget_signal[name].changeReref)
            self.comboBox_spectrogram.clear()
            self.comboBox_spectrogram.addItems(stream_configs[name]['channel_names'] + ['All'])
            self.comboBox_spectrogram.setCurrentIndex(stream_configs[name]['channel_count'])

    def bandPass(self, verbose=True):
        low = float(self.lineEdit_bandpass_low.text())
        high = float(self.lineEdit_bandpass_high.text())
        self.widget_signal[self.streamname].init_bandpass_filter(low, high, name=self.streamname, verbose=verbose)

    def bandStop(self, verbose=True):
        low = float(self.lineEdit_bandstop_low.text())
        high = float(self.lineEdit_bandstop_high.text())
        status = self.widget_signal[self.streamname].init_bandstop_filter(low, high, name=self.streamname, verbose=verbose)
        if status:
            self.label_status.setText(f"bandstop is ON")
        else:
            self.label_status.setText(f"bandstop is OFF")

    def switch_asr(self):
        ON = self.widget_signal[self.streamname].switch_asr()
        if ON:
            self.label_status.setText(f"ASR is ON")
        else:
            self.label_status.setText(f"ASR is OFF")

    def organizeSignalWidget(self):
        self.wid = QtGui.QWidget(self) # ref: https://stackoverflow.com/questions/37304684/qwidgetsetlayout-attempting-to-set-qlayout-on-mainwindow-which-already
        self.setCentralWidget(self.wid)
        ## handle SignalWidget layout
        vLayout = [QVBoxLayout()]
        vLayout[0].setSpacing(0)
        vLayout[0].setContentsMargins(0, 0, 0, 0)
        col_idx = 0
        row_idx = 0
        cur_channel_count = 0
        for sn in target_streams: # iterate in this way to let user decides how to arrange stream layout
            if sn in self.widget_signal:
                w = self.widget_signal[sn]
            else:
                continue
        # for w in self.widget_signal.values():
            if cur_channel_count != 0 and cur_channel_count + w.total_ch > channel_per_column:
                col_idx += 1
                row_idx = 0
                cur_channel_count = 0
                vLayout.append(QVBoxLayout())
                vLayout[-1].setSpacing(0)
                vLayout[-1].setContentsMargins(0, 0, 0, 0)
            cur_channel_count += w.total_ch
            vLayout[col_idx].addWidget(w, stretch=stream_configs[w.streamname]['stretch_factor'])
            row_idx += 1

        plotLayout = QHBoxLayout()
        plotLayout.setSpacing(0)
        plotLayout.setContentsMargins(0, 0, 0, 0)
        for i, vl in enumerate(vLayout):
            plotLayout.addLayout(vl, stretch=1)

        ## Handle control panel layout
        panel_layout = QHBoxLayout()
        panel_layout.addWidget(self.groupBox_ctrl, stretch=5)
        panel_layout.addWidget(self.groupBox_record, stretch=4)
        panel_layout.addWidget(self.groupBox_process, stretch=2)
        panel_layout.addWidget(self.groupBox_plot, stretch=2)
        panel_layout.addWidget(self.groupBox_spectrogram, stretch=2)
        panel_layout.addWidget(self.groupBox_ASR, stretch=1)

        self.layout = QVBoxLayout()
        plot_stretch, panel_strech = self.getStrecchFactors()
        self.layout.addLayout(plotLayout, stretch=plot_stretch)
        self.layout.addLayout(panel_layout, stretch=panel_strech)
        self.wid.setLayout(self.layout)
        if self.geometry().height() < 300:
            self.resize(self.geometry().width(), 600)
        self.resizeEvent(event=None)

    def getStrecchFactors(self):
        panel_height = 100.0
        scale = self.geometry().height()/panel_height
        factor = (self.geometry().height()-panel_height)/panel_height
        if scale < 2:
            plot_stretch = 0
            panel_strech = 100
        else:
            plot_stretch = round(factor)
            panel_strech = 1
        return plot_stretch, panel_strech

    def addNewPlotWidget(self, stream):
        name = stream.name()
        self.widget_signal[name] = SignalPlotWidget(parent=self, inlets=[], plot_duration=self.plot_duration, spacing=stream_configs[name]['spacing'], plot_mode=self.comboBox_mode.currentText(),
                                        streamname=name, stream_config=stream_configs[name])
        if stream_configs[name]['add_legend']:
            self.widget_signal[name].plt.addLegend()
        self.widget_signal[name].saveName = self.lineEdit_savename.text()[:-4] + f'_{name}.csv'
        self.widget_signal[name].setEventFile(prefix=self.lineEdit_savename.text()[:-4]) # set for all, just in case
        if name in stream_configs:
            name_channel = stream_configs[name]['channel_names']
        else:
            name_channel = [f'Ch{i}' for i in range(stream.channel_count())]
        self.widget_signal[name].inlets = [DataInlet(stream, self.widget_signal[name].plt, self.plot_duration, self.global_clock_ref, name_channel, ch_colors)]
        self.widget_signal[name].srate = self.widget_signal[name].inlets[0].srate
        self.widget_signal[name].total_ch = self.widget_signal[name].inlets[0].channel_count
        self.n_inlet += 1
        # create a timer that will move the view every update_interval ms
        self.widget_signal[name].update_timer.start(update_interval)
        # create a timer that will pull and add new data occasionally
        self.widget_signal[name].pull_timer.start(pull_interval)
        self.widget_signal[name].show()

    def addSpectrogram(self):
        win_size = int(self.lineEdit_winsize.text())
        step_size = int(self.lineEdit_stepsize.text())
        ch_idx = self.comboBox_spectrogram.currentIndex()
        if ch_idx == stream_configs[self.streamname]['channel_count']:
            n_row = 2
            n_col = 4
            size = self.screen.size()
            height = size.height()//n_row
            width = size.width()//n_col
            for c in range(stream_configs[self.streamname]['channel_count']):
                self.widget_signal[self.streamname].addSpectrogram(c, win_size, step_size, QtCore.Qt.FramelessWindowHint)
                self.widget_signal[self.streamname].spectrogram[-1].resize(width, height)
                self.widget_signal[self.streamname].spectrogram[-1].move(c%n_col*width, c//n_col*height)
        else:
            self.widget_signal[self.streamname].addSpectrogram(ch_idx, win_size, step_size)

    def changePauseVariable(self):
        for w in self.widget_signal.values():
            w.changePauseVariable()

    def calibrate_asr(self):
        self.widget_signal[self.streamname].calibrate_asr()
        self.label_status.setText("ASR calibrated and ON")

    def init_buttons(self):
        self.pushButton_connectLSL.clicked.connect(self.connect_to_LSL)
        self.pushButton_pause.clicked.connect(self.changePauseVariable)
        self.pushButton_record.clicked.connect(self.saveData)
        self.pushButton_stop_record.clicked.connect(self.stopSave)
        self.pushButton_filter.clicked.connect(self.bandPass)
        self.pushButton_filter2.clicked.connect(self.bandStop)
        self.pushButton_spectrogram.clicked.connect(self.addSpectrogram)
        self.pushButton_file.clicked.connect(self.getfile)
        self.pushButton_ASR.clicked.connect(self.calibrate_asr)
        self.pushButton_ASR_switch.clicked.connect(self.switch_asr)


        self.comboBox_mode.currentTextChanged.connect(self.changePlotMode)
        self.comboBox_streamName.currentTextChanged.connect(self.changeStreamName)

        self.lineEdit_duration.editingFinished.connect(self.changePlotDuration)
        self.lineEdit_spacing.editingFinished.connect(self.changeSpacing)

def main():
    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    fontsize = read_ui_config()
    app.setStyleSheet("QLabel{font-size:" + fontsize + ";}")
    app.setStyleSheet("QPushButton{font-size:" + fontsize + ";}")
    # app.setStyleSheet("QLabel{font-size: 6.5pt;}")
    # app.setStyleSheet("QPushButton{font-size: 6.5pt;}")

    screen = app.primaryScreen()
    main = MainWindow(screen)
    # main.showMaximized()
    main.showNormal()
    main.move(0, 0)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()