import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def main(cfg):

    # load filenames

    # for month in months:
        # extract months  
        # if not obs :
            # calculate sia
            # calculate anos 
        # annual stats (needed to get the same timestamp)
        # if model, calculate mme stats
        # plot timeseries 
    
    # plot meshplot for obs and CMIP6 

    logger.info('Succes')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)