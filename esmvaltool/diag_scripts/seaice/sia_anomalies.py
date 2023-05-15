import iris
import logging
import numpy as np
from pathlib import Path

import esmvaltool.diag_scripts.shared.plot as eplot
import esmvalcore.preprocessor as eprep
from esmvaltool.diag_scripts.shared import run_diagnostic, group_metadata, save_data, save_figure

logger = logging.getLogger(Path(__file__).stem)

def get_provenance_record(ancestor_files):

    record = {
        'authors': ['malinina_elizaveta'],
        'ancestors': ancestor_files,
    }
    return record

def calculate_sia(cube):

    area = iris.analysis.cartography.area_weights(cube, normalize=False)
    time = cube.coord('time')

    siconc = cube.data / 100 # since the data is in %, it has to be converted into fraction
    area = np.ma.array(area, mask=cube.data.mask)

    sia_arr = siconc * area

    sia = (sia_arr.sum(axis=(1, 2)) / (1000 ** 2)) / 1000000 # iris provides area in m, converting it to 10^6km2
    # for now passing parent cube attributes
    sia_cube = iris.cube.Cube(sia, standard_name='sea_ice_area', long_name='sea ice area', var_name='siarea',
                                units="1e6 km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])   

    return sia_cube

def main(cfg):

    input_data = cfg['input_data'].values()
    f_names = group_metadata(input_data, 'filename')

    for f_name in f_names.keys():
        prov_record = get_provenance_record([f_name])
        logger.info("Processing dataset %s", f_name)
        cube = iris.load_cube(f_name)
        logger.info("Calculating sea ice area (SIA) in %s", f_name)
        sia_cube = calculate_sia(cube)
        logger.info("Calculating sea ice area (SIA) anomalies for %s", f_name)
        sia_ano_cube = eprep.anomalies(sia_cube,  'full', 
                                                reference = {'start_year': cfg['anomaly_base_period'][0], 
                                                'start_month': 1, 'start_day':1, 
                                                'end_year': cfg['anomaly_base_period'][1], 
                                                'end_month': 12, 'end_day':31} )
        logger.info("Saving sea ice area (SIA) for %s", f_name)
        base_name = f_names[f_name][0]['project'] +'_' + f_names[f_name][0]['alias'] + '_sia_anomaly'
        save_data(base_name, prov_record, cfg, sia_ano_cube)
        logger.info("Plotting sea ice area (SIA) anomalies for %s", f_name)
        eplot.quickplot(sia_ano_cube, 'plot')
        logger.info("Saving sea ice area (SIA) anomalies figure for %s", f_name)
        save_figure(base_name, prov_record, cfg)
        logger.info("Success")

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
