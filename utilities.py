import os
import sys
import errno
import subprocess
import json
import logging

def python_mkdir(directory):
    '''A function to make a unix directory as well as subdirectories'''
    try:
        if directory: os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else: raise

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def runCommand(command):
    logging.debug(command)
    return subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0]

def get_das(query,verbose=False):
    command = 'dasgoclient -query "{}"'.format(query)
    result = runCommand(command)
    if verbose: logging.debug(result)
    return [line.decode('utf-8') for line in result.split()]

def load(fname):
    jname = '{}.json'.format(fname)
    if not os.path.exists(jname): return {}
    with open(jname) as f:
        return json.load(f)

def dump(fname,content):
    jname = '{}.json'.format(fname)
    python_mkdir(os.path.dirname(jname))
    with open(jname,'w') as f:
        json.dump(content,f, indent=4, sort_keys=True)
