# Utility functions

from __future__ import print_function

import os
import os.path as path


def create_directories(dirs):
    '''
        Create directory rooted at cwd for every entry in dirs.
        Return list of full paths to created directories
    '''
    cwd = os.getcwd()
    result = []

    for d in dirs:
        dir_path = path.join(cwd, d)
        if not path.exists(dir_path): os.mkdir(dir_path)
        result.append(dir_path)

    return result
