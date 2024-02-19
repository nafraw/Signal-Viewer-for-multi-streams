#!/usr/bin/env python
"""
ReceiveAndPlot example for LSL
This example shows data from all found outlets in realtime.
It illustrates the following use cases:
- efficiently pulling data, re-using buffers
- automatically discarding older samples
- online postprocessing
"""

import numpy as np
import math
import pylsl
import pyqtgraph as pg

class Inlet:
    """Base class to represent a plottable inlet"""
    def __init__(self, info: pylsl.StreamInfo, plot_duration, global_clock_ref):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()
        self.srate = info.nominal_srate()
        self.global_clock_ref = global_clock_ref

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem, plot_duration, global_clock_ref, curve_names, ch_colors):
        super().__init__(info, plot_duration, global_clock_ref)
        # calculate the size for our buffer, i.e. two times the displayed data
        self.plot_samples = math.ceil(info.nominal_srate() * plot_duration)
        self.plot_duration = plot_duration
        self.srate = info.nominal_srate()
        bufsize = (4 * self.plot_samples, info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        self.timestamps = None
        empty = np.array([])
        # create one curve object for each channel/line that will handle displaying the data
        self.curves = [pg.PlotCurveItem(x=np.array([x for x in range(self.plot_samples)]), y=np.full(self.plot_samples, fill_value=np.nan),
                            name=curve_names[i], autoDownsample=True, pen=pg.mkPen(pg.mkColor(ch_colors[i%len(ch_colors)]))) for i in range(self.channel_count)]
        for curve in self.curves:
            plt.addItem(curve)
        self.prev_x = 0 # for the x position of plotting
        self.ref_time_at_left = pylsl.local_clock()

    def pull_and_plot(self, plot_time, yshift, proc_func=None, plot_mode='Scroll'):
        # pull the data
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        self.timestamps = ts
        self.chunk_size = len(ts)
        if ts:
            ts = np.asarray(ts)
            self.chunk_size = ts.size
            y = self.buffer[0:ts.size, :].copy()
            if proc_func is not None:
                for f in proc_func:
                    y = f(y)
            if plot_mode == 'Scroll':
                self.scroll_data(y, ts, yshift, plot_time)
            else:
                self.refresh_data(y, ts, yshift, plot_time)
            return y # return processed y

    def refresh_data(self, y, ts, yshift, plot_time):
        n_start_over = self.prev_x + ts.size - self.plot_samples # > 0 = need to go back
        this_x, _ = self.curves[0].getData()
        if len(this_x) < self.plot_samples:
            this_x = np.arange(min(self.prev_x + ts.size, self.plot_samples))
        for ch_ix in range(self.channel_count):
            y[:, ch_ix] = y[:, ch_ix] + yshift[ch_ix]
            _, this_y = self.curves[ch_ix].getData()
            if n_start_over <= 0:
                if len(this_y) == self.plot_samples:
                    this_y[self.prev_x:self.prev_x+ts.size] = y[:, ch_ix]
                else:
                    this_y = np.hstack((this_y, y[:, ch_ix]))
            else:
                this_y[0:n_start_over] = y[ts.size-n_start_over:, ch_ix]
                if len(this_y) != self.plot_samples:
                    old_y = this_y
                    this_y = np.arange(self.plot_samples)
                    this_y[:self.prev_x] = old_y
                this_y[self.prev_x:self.plot_samples] = y[0:ts.size-n_start_over, ch_ix]
            self.curves[ch_ix].setData(x=this_x, y=this_y)

        if n_start_over > 0:
            self.prev_x = n_start_over
        else:
            self.prev_x += ts.size
        self.ref_time_at_left = ts[-1] - (self.prev_x-1)/self.srate - self.global_clock_ref

    def scroll_data(self, y, ts, yshift, plot_time):
        this_x = None
        old_offset = 0
        new_offset = 0
        for ch_ix in range(0, self.channel_count):
            # y[:, ch_ix] = y[:, ch_ix] + yshift[ch_ix] - np.mean(y[:, ch_ix])
            y[:, ch_ix] = y[:, ch_ix] + yshift[ch_ix]
            # we don't pull an entire screen's worth of data, so we have to
            # trim the old data and append the new data to it
            old_x, old_y = self.curves[ch_ix].getData()
            # the timestamps are identical for all channels, so we need to do
            # this calculation only once
            if ch_ix == 0:
                # find the index of the first sample that's still visible,
                # i.e. newer than the left border of the plot
                old_offset = old_x.searchsorted(plot_time)
                # same for the new data, in case we pulled more data than
                # can be shown at once
                new_offset = ts.searchsorted(plot_time)
                # append new timestamps to the trimmed old timestamps
                this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
            # append new data to the trimmed old data
            this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch_ix]))
            self.curves[ch_ix].setData(this_x, this_y)

class MarkerInlet(Inlet):
    """A MarkerInlet shows events that happen sporadically as vertical lines"""
    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem, plot_duration, global_clock_ref, remove_limit_range=100):
        super().__init__(info, plot_duration, global_clock_ref)
        self.items = []
        self.plt = plt
        self.plot_dur = plot_duration
        self.new_events = {'time': None, 'event': None}
        self.remove_limit_range = remove_limit_range # in unit of samples, too small will make events hard to remove in the refresh mode
        self.refresh_time = 0
        self.prev_check_time = pylsl.local_clock()

    def pull_and_plot(self, plot_time, time_left_refresh, idx_prev_x_signal, srate=None, plot_mode='Scroll', prev_x=0, sname=None):
        # note: plot_time is the time at left most part
        # srate is needed for refresh mode

        strings, timestamps = self.inlet.pull_chunk(0)
        self.new_events['time'] = timestamps
        self.new_events['event'] = strings

        if not timestamps and ((pylsl.local_clock() - self.prev_check_time) > 1):
            self.refresh_time = idx_prev_x_signal

        # add new events
        if timestamps:
            # update refresh_time because it may happens that event marker has a much higher sampling rate for a very slow signal
            for string, ts in zip(strings, timestamps):
                if isinstance(string[0], str): # this should be sent by LSL from the trigger_manager
                    string = string[0]
                else:
                    string = str(string) # force as a string
                if plot_mode == 'Refresh':
                    ts_ = ts
                    ts = ts - self.global_clock_ref
                    if ts >= time_left_refresh:
                        ts = (ts - time_left_refresh)*srate
                        if ts > (self.plot_dur):
                            ts -= (self.plot_dur)
                    else:
                        ts = (self.plot_dur - (time_left_refresh - ts))*srate

                    infline = pg.InfiniteLine(ts, angle=90, movable=False, label=string)
                    if ts_ == timestamps[-1]:
                        # print(f'{sname}, refresh: {ts}')
                        self.refresh_time  = ts
                        self.prev_check_time = pylsl.local_clock()
                else:
                    infline = pg.InfiniteLine(ts, angle=90, movable=False, label=string)

                infline.label.setHtml(f'<font size="+4">{string}</font>')
                self.items.append(infline)
                self.plt.addItem(self.items[-1])
        # remove old events
        nData = self.plt.getAxis('bottom').range[1] - self.plt.getAxis('bottom').range[0]
        remove_limit = self.refresh_time + self.remove_limit_range
        # if remove_limit > nData: remove_limit -= nData
        checked = 0
        while True:
            if len(self.items) == 0: break
            if len(self.items) == checked: break
            evt_x = self.items[checked].getXPos()
            if plot_mode == 'Scroll':
                if evt_x < plot_time:
                    self.plt.removeItem(self.items[checked])
                    self.items.pop(checked)
                else:
                    checked+=1
                    break
            if plot_mode == 'Refresh':
                if (self.refresh_time < evt_x < remove_limit) or \
                    ((remove_limit > nData) and evt_x < (remove_limit-nData)):
                    self.plt.removeItem(self.items[checked])
                    self.items.pop(checked)
                else:
                    checked+=1

    def clear_all_markers(self):
        for m in reversed(self.items):
            self.plt.removeItem(m)

class dataBuffer1D():
    def __init__(self, bufsize, init_value=np.nan):
        self.buffer = np.full(shape=bufsize, fill_value=init_value)
        self.ptr = 0
        self.bufsize = bufsize

    def addData(self, data):
        n_input = data.shape[0]
        n_overflow = self.ptr + n_input - self.bufsize
        if n_overflow > 0:
            self.pushData(n_overflow)
            self.buffer[self.ptr:] = data
            self.ptr = self.bufsize
        else:
            self.buffer[self.ptr: self.ptr+n_input] = data
            self.ptr += n_input

    def fetchData(self, size, push_size=None):
        rt_size = 0
        data = None
        if self.ptr > size:
            rt_size = size
            data = self.buffer[:size].copy()
            if push_size is None: push_size = size
            self.pushData(push_size)
        return data, rt_size

    def pushData(self, size, clear=True):
        self.buffer[0:self.bufsize-size] = self.buffer[size:]
        if clear:
            self.buffer[self.bufsize-size:] = np.nan
        self.ptr -= size

class dataBuffer2D():
    def __init__(self, shape, init_value=np.nan):
        self.buffer = np.full(shape=shape, fill_value=init_value)
        self.ptr = 0
        self.bufsize = shape[0]

    def addData(self, data):
        n_input = data.shape[0]
        n_overflow = self.ptr + n_input - self.bufsize
        if n_overflow > 0:
            self.pushData(n_overflow)
            self.buffer[self.ptr:, :] = data
            self.ptr = self.bufsize
        else:
            self.buffer[self.ptr: self.ptr+n_input, :] = data
            self.ptr += n_input

    def fetchData(self, size, push_size=None):
        rt_size = 0
        data = None
        if self.ptr > size:
            rt_size = size
            data = self.buffer[:size, :].copy()
            if push_size is None: push_size = size
            self.pushData(push_size)
        return data, rt_size

    def pushData(self, size, clear=True):
        self.buffer[0:self.bufsize-size, :] = self.buffer[size:, :]
        if clear:
            self.buffer[self.bufsize-size:, :] = np.nan
        self.ptr -= size