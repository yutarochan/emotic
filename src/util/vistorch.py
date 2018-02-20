'''
Visdom Pytorch
Log Visualization and Progress Tracking Utility
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import visdom

class Vistorch:
    def __init__(self, log_title, console_out=True, port=8097):
        self.vis = visdom.Visdom()
        
        # Text Log Setup
        self.textlog = self.vis.text('<b>'+log_title+'</b><br>')
        self.console_out = console_out

        # Plot Containers
        self.plots = dict()

    def print_log(self, data):
        self.vis.text(data, win=self.textlog, append=True)
        if self.console_out: print(data)

    def plot_line(self, data, name):
        if name not in plots: 

if __name__ == '__main__':
    log = Vistorch('Vistorch Logger')
    log.print_log('This is a test!')
    log.print_log('This is so damn cool!')
