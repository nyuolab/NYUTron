import logging
import contextlib
import os
import sys
import datetime as dt
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', default=False, action='store_true')
parser.add_argument('--stop', default=False, action='store_true')

class Std2File(object):
    """
    Redirect stoout and stderr to a local file.
    """
    #Class vars
    stdout = None
    stderr = None
    fd = None

    def __init__(self, f, std):
        self.std = std
        self.f = f

    @staticmethod
    def enable(f='./log/nytdev.log'):
        if Std2File.stdout is None:
            Std2File.stdout = sys.stdout
            Std2File.stderr = sys.stderr
            Std2File.fd = open(f, 'a+')
            sys.stdout = Std2File(Std2File.fd, sys.stdout)
            sys.stderr = Std2File(Std2File.fd, sys.stderr)
        print('\n\nbackup stdout/stderr to %s at %s\n' % (f, dt.datetime.now()))

    @staticmethod
    def disable():
        if Std2File.stdout is not None:
            sys.stdout = Std2File.stdout
            sys.stderr = Std2File.stderr
            Std2File.stderr = Std2File.stdout = None
            Std2File.fd.close()
            Std2File.fd = None

    def __getattr__(self, name):
        return object.__getattribute__(self.f, name)

    def write(self, x):
        self.std.write(x)
        self.std.flush()
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.std.flush()
        self.f.flush()


def main():
    args = parser.parse_args()
    if args.start:
        Std2File.enable("./log/nytlogs.log")

    if args.stop:
        Std2File.disable()


if __name__ == '__main__':
    main()