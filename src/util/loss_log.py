'''
Loss Logger
Utility function for persisting loss values onto log.
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os.path

def write(path, data):
    output = open(path, 'a' if os.path.isfile(path) else 'w')
    output.write(data + '\n')
    output.close()
