"""
Helper file for methods needed aside
"""

import time
import os
import shutil
import matplotlib.pyplot as plt

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)


def write_run_results_2file(file_path, exp_name, results_dict):
    with open(file=file_path, mode='a+') as f:
        f.write('{}:\n'.format(exp_name))
        for metric, value in results_dict.items():
            f.write('{0: 2d}: {1: 6d}\n'.format(metric, value))


def plot_image(image, img_shape, file_name):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.savefig('{}.png'.format(file_name))