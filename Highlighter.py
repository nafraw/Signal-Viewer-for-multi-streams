# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:46:14 2021

@author: Ping-Keng Jao
"""
# Purpose of this file:
#   defines frequently used warn and error message functions to highlight texts
#   add personal formats of highlighting messages if desired

# Some relevant references below
# ref: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
# ref: note: the best answer did not work

from termcolor import colored, COLORS, HIGHLIGHTS, ATTRIBUTES

class Highlighter:
    ''' The following three variables are together refered as format here '''
    TEXT_COLOR = {'WARN': None, 'ERROR': None, 'IMPORTANT': None}
    BACK_COLOR = {'WARN': 'on_magenta', 'ERROR': 'on_red', 'IMPORTANT': 'on_green'}
    STYLE = {'WARN': None, 'ERROR': None, 'IMPORTANT': None}
    
    def __init__(self):
        pass
    
    ''' Functions to Print colored texts '''
    def custom_format(name_of_format: str, msg: str, show=True):
        # print(Highlighter.TEXT_COLOR[name_of_format], Highlighter.BACK_COLOR[name_of_format])
        msg = colored(msg, color=Highlighter.TEXT_COLOR[name_of_format], on_color=Highlighter.BACK_COLOR[name_of_format], attrs=Highlighter.STYLE[name_of_format])
        if show: print(msg)
        return msg        
    
    def error(msg: str, show=True) -> str:
        return Highlighter.custom_format('ERROR', msg, show)
    
    def warn(msg: str, show=True) -> str:
        return Highlighter.custom_format('WARN', msg, show)
    
    def important(msg: str, show=True) -> str:
        return Highlighter.custom_format('IMPORTANT', msg, show)
    
    ''' Functions to Print supported colors and styles '''
    def show_available_text_colors():
        print(COLORS.keys())

    def show_available_background_colors():
        print(HIGHLIGHTS.keys())
    
    def show_available_styles():
        print(ATTRIBUTES.keys())
    
    ''' Functions to Change formats '''    
    def set_format(name_of_format: str, color=None, back_color=None, style=None):        
        if color:
            if color not in COLORS:
                Highlighter.error(f"{color} doesn't exist, please call Highlighter.show_available_text_colors() for a supported list")
            Highlighter.TEXT_COLOR[name_of_format] = color
        else:
            Highlighter.TEXT_COLOR[name_of_format] = None
        if back_color:
            if back_color not in HIGHLIGHTS:
                Highlighter.error(f"{back_color} doesn't exist, please call Highlighter.show_available_back_colors() for a supported list")
            Highlighter.BACK_COLOR[name_of_format] = back_color
        else:
            Highlighter.BACK_COLOR[name_of_format] = None
        if style:
            if not isinstance(style, list):
                Highlighter.error("The style argument must be a list")
            for s in style:
                if s not in ATTRIBUTES:
                    Highlighter.error(f"{s} doesn't exist, please call Highlighter.show_available_styles() for a supported list")
            Highlighter.STYLE[name_of_format] = style
        else:
            Highlighter.STYLE[name_of_format] = None
        
    def reset_default_color(): # only reset default colors of WARN and ERROR
        Highlighter.TEXT_COLOR['WARN'] = None
        Highlighter.BACK_COLOR['WARN'] = 'on_magenta'
        Highlighter.STYLE['WARN'] = None
        Highlighter.TEXT_COLOR['ERROR'] = None
        Highlighter.BACK_COLOR['ERROR'] = 'on_red'
        Highlighter.STYLE['ERROR'] = None
    
    def reset_default_format(): # remove all personalized formats
        Highlighter.TEXT_COLOR = {'WARN': None, 'ERROR': None}
        Highlighter.BACK_COLOR = {'WARN': 'on_magenta', 'ERROR': 'on_red'}
        Highlighter.STYLE = {'WARN': None, 'ERROR': None}
    
    
    
if __name__ == '__main__':
    Highlighter.warn('warn')
    Highlighter.error('error')
    a = 3
    msg = f"{a}+b"
    Highlighter.set_format('format_1', color='green', back_color='on_yellow', style=['bold', 'blink'])
    Highlighter.custom_format('format_1', 'format_1 now looks like' + msg)
    Highlighter.set_format('WARN', color='green', back_color=None, style=['bold', 'blink'])
    Highlighter.warn('New warn')
    Highlighter.reset_default_color()
    print('reset warning message color')
    Highlighter.warn(msg)
    Highlighter.set_format('format_1', color='green', back_color='on_red', style=['test']) # output error message due to wrong input
    Highlighter.set_format('format_1', color='green', back_color='on_yes', style=None) # output error message due to wrong input
    Highlighter.show_available_text_colors()