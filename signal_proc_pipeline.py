# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:16:31 2021

@author: Ping-Keng Jao
"""

import numpy as np
import scipy.signal as sig
import pyqtgraph as pg
from Highlighter import Highlighter as hmsg



def reref(x, idx):
    if idx==0: # no re-reference
        y = x
    elif idx > x.shape[1]: # CAR: common average re-referencing
        ref = np.mean(x, axis=1)
        y = x - ref[:, np.newaxis]
    else: # single reference, -1 because 0 is reserved for no referencing
        y = x - x[:, [idx-1]]
    return y

class filtClass():
    def __init__(self, b, a, n_ch):
        self.b=b
        self.a=a
        self.z = sig.lfilter_zi(self.b, self.a)*0
        self.z = self.z[:, np.newaxis]
        self.z = np.repeat(self.z, n_ch, axis = 1)
        # print(self.z.shape)

    def filter(self, x, NaN_as_zero=True): # x is time-by-channel
        if NaN_as_zero:
            x[np.isnan(x)] = 0
        fx, self.z = sig.lfilter(self.b, self.a, x, zi=self.z, axis=0)
        return fx

def plot_frequency_response(fs, b, a=1, plt=None):
    freq, h = sig.freqz(b, a, fs=fs)
    # fig, ax1 = plt.subplots()
    plt.setTitle('Digital filter frequency response')
    plt.setLabel("left", 'Amplitude [dB]', color='gray')
    plt.setLabel("bottom", 'Frequency Hz', color='gray')
    plt.plot(freq, get_db(h), linecolor='red')

def create_fir_filter(fs, cutoff, nTap, filter_type=None, verbose=True):
    # nTap: str such as '1s' or '1000ms' or int > 0
    if filter_type is None:
        if len(cutoff) > 1:
            filter_type = 'bandpass'
            hmsg.important('filter_type not specified, assuming bandpass')
        else:
            filter_type = 'lowpass'
            hmsg.important('filter_type not specified, assuming lowpass')
    if isinstance(nTap, str):
        nTap = parse_duration(duration=nTap, fs=fs)
    b = sig.firwin(cutoff=cutoff, fs=fs, numtaps=nTap, pass_zero=filter_type)
    check_attenuation(fs, cutoff, b)
    if verbose:
        pw = pg.plot(title='LSL Plot')
        plt = pw.getPlotItem()
        plt.enableAutoRange(x=True, y=True)
        plot_frequency_response(fs, b, plt=plt)
    return b


def create_iir_filter(fs, cutoff, order, filter_type='bandpass', ftype='butter', title='LSL Plot', verbose=True):
    b, a= sig.iirfilter(fs=fs, N=order, Wn=cutoff, btype=filter_type, ftype=ftype, output='ba')
    check_attenuation(fs, cutoff, b, a)
    stable = np.all(np.abs(np.roots(a))<1) # ref: https://stackoverflow.com/questions/8811518/scipy-lfilter-returns-only-nans
    if not stable:
        hmsg.warn('Designed IIR filter is not stable!!!')
    if verbose:
        pw = pg.plot(title=title)
        plt = pw.getPlotItem()
        plt.enableAutoRange(x=True, y=True)
        plot_frequency_response(fs, b, a, plt)
    return b, a

def get_db(v):
    return 20 * np.log10(abs(v))

def check_attenuation(fs, cutoff, b, a=1, meet_db=-12): #TODO: check if 3dB attenuation is met
    freq, h = sig.freqz(b, a, fs=fs)
    mag_db = get_db(h)
    for f in cutoff:
        # has the exact frequency?
        idx = np.where(freq==f)
        if idx[0].size != 0:
            if mag_db[idx[0]] > meet_db:
                hmsg.warn(f'attenuation at {f} Hz is {mag_db[idx[0]]}dB > {meet_db}dB')
            continue
        # use interpolate for the response at freq f.
        idx = np.where(freq<f)
        if idx[0].size == 0:
            hmsg.error('Something went wrong in check_attenuation...please debug')
        target_bin = (idx[0][-1], idx[0][-1]+1)
        atten_at_f = mag_db[target_bin[0]] + (mag_db[target_bin[1]] - mag_db[target_bin[0]]) * (f - freq[target_bin[0]]) / (freq[target_bin[1]] - freq[target_bin[0]])
        if atten_at_f > meet_db:
            hmsg.warn((f'attenuation at {f} Hz is {atten_at_f}dB > {meet_db}dB'))

def parse_duration(duration, fs):
    assert(duration[-1]=='s')
    dur = duration[:-1]
    multiplier = 1
    if not str.isdigit(duration[-2]):
        if duration[-2] == 'u':
            multiplier = 10**6
        elif duration[-2] == 'm':
            multiplier = 10**3
        else:
            hmsg.error(f'cannot parse {duration}')
        dur = dur[:-1]
    return int(float(dur)*fs*multiplier)

if __name__ == '__main__':
    fs=250
    cutoff = [0.5, 40]
    b = create_fir_filter(fs=fs, cutoff=[0.5, 40], nTap='1s', filter_type='bandpass', verbose=False)
    b, a = create_iir_filter(fs=fs, cutoff=[0.5, 40], order=4, filter_type='bandpass', ftype='butter', verbose=False)
    data = np.ones([1000, 2])
    out = sig.lfilter(b, a, data, axis=0)
