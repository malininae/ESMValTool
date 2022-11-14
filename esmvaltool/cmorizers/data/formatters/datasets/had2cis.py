"""ESMValTool CMORizer for Had2CIS data.

Tier
    Tier 2: other freely available datasets.

Source
    https://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/TMP/

Last access
    20221110

Download and processing instructions
    Just download the file following the link
"""

import logging
import os

import iris
import cftime
from cf_units import Unit
import numpy as np

from esmvaltool.cmorizers.data import utilities as utils

logger = logging.getLogger(__name__)

def _fix_time_coord(cube):
    """Convert time from the integer 'yyyymmdd' into 
        real gregorian datetime units.""" 

    raw_time_pnts = cube.coord('time').points
    upd_time_pnts = []

    for raw_time_pnt in raw_time_pnts:
        raw_str_tp = str(int(raw_time_pnt))
        y = int(raw_str_tp[:4])
        m = int(raw_str_tp[4:6])
        d = 15 # setting to 15th, beacause data is monthly
        upd_time_pnt = cftime.datetime(y, m, d, calendar='gregorian')
        upd_time_pnts.append(upd_time_pnt)

    upd_time_pnts = np.asarray(upd_time_pnts)
    upd_time_nums = cftime.date2num(upd_time_pnts, 'days since 1950-01-01 00:00:00', calendar= 'gregorian') 
    time_unit  = Unit('days since 1950-01-01 00:00:00', calendar= 'gregorian')

    cube.coord('time').points = upd_time_nums
    cube.coord('time').units = time_unit        

    return cube


def _extract_variable(var, var_info, cmor_info, attrs, filepath, out_dir):
    """"Extract variable."""

    var = cmor_info.short_name
    raw_cube = iris.load_cube(filepath)

    # converting time
    cube = _fix_time_coord(raw_cube)

    # convert fraction to %
    sic_cube = cube * 100

    utils.fix_var_metadata(sic_cube, cmor_info)
    utils.convert_timeunits(sic_cube, 1950)
    utils.set_global_atts(sic_cube, attrs)
    utils.save_variable(sic_cube,
                        var,
                        out_dir,
                        attrs,
                        unlimited_dimensions=['time'])

    
def cmorization (in_dir, out_dir, cfg, cfg_user, start_date, end_date):
    """Cmorization function call"""
    glob_attrs = cfg['attributes']
    cmor_table = cfg['cmor_table']
    filename = cfg['filename']

    filepath = os.path.join(in_dir, filename)

    if os.path.isfile(filepath):
        logger.info("Found input file '%s'", filepath)
    else: 
        raise OSError(f"Cannot find input file '{filename}' in '{in_dir}'")
    

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        logger.info("CMORizing variable: %s", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        _extract_variable(var, var_info, cmor_info, glob_attrs, filepath,
                          out_dir)