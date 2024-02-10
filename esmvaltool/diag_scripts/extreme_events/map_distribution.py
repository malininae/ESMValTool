import iris
import iris.plot as iplt
from iris.time import PartialDateTime
import cartopy.crs as ccrs 
import cartopy.feature as cff
import geopandas as gpd 
import esmvalcore.preprocessor as eprep
from datetime import datetime, timedelta, date
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def extract_partial_dates(cube: iris.cube.Cube, inp_date: date, window: int):
    '''
    This function extracts the dates in +/- half window in each year.

    Input:
        cube: iris cube from which the data will be extracted
        inp_date: date on which the extraction is centered
        window: the half_window for extraction around the inp_date
    Output:
        short_cb: iris cube with extracted dates
    '''
    min_bord = inp_date - timedelta(days=window)
    max_bord = inp_date + timedelta(days=window)
    min_constr = PartialDateTime(month=min_bord.month, day=min_bord.day)
    max_constr = PartialDateTime(month=max_bord.month, day=max_bord.day)
    pdt_constr = iris.Constraint(
                    time = lambda cell: min_constr <= cell.point <= max_constr)
    short_cb = cube.extract(pdt_constr)

    return short_cb

def plot_map(cube: iris.cube.Cube, dataset: str, cfg: dict):
    '''
    This function plots map

    Input:
        cube: lat/lon resolved iris cube to be plotted
        cfg: standard ESMValTool config dictionary
    '''
    # loading matplotlib style saved in ESMValTool folder
    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    shape_file_path = os.path.join(cfg['auxiliary_data_dir'], 
                                                            cfg['shape_file'])
    shape_gpd = gpd.read_file(shape_file_path)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, 
                           subplot_kw={'projection': ccrs.LambertConformal(
                                            central_longitude=-91.87,
                                            standard_parallels=(49,76))})
    fig.set_size_inches(7., 7.)
    fig.set_dpi(200)

    max_t = cube.collapsed(['latitude', 'longitude'], iris.analysis.MAX).data
    min_t = cube.collapsed(['latitude', 'longitude'], iris.analysis.MIN).data
    bord = np.floor(np.abs(np.asarray([max_t, min_t]).flatten()).max()*0.975)
    ax.coastlines(linewidth=0.5)
    map_pl = iplt.contourf(cube, axes=ax, levels=np.arange(-bord, bord+1,1), 
                                                cmap='coolwarm', extend='both')
    # add title
    # add colorbar 
    # adjust the location on the figure
    # add the legend for the regions
    shape_gpd.boundary.plot(color=None, edgecolor='grey', linewidth=0.5, ax=ax, 
                                                  transform=ccrs.PlateCarree())
    ax.set_extent([-60.5, -120.5,37.8,82.8])    
    fig.savefig(os.path.join(cfg['plot_dir'], 
                                f'map_canada_{dataset}.'+cfg['output_file_type']))

    return

def main(cfg):

    input_data = cfg['input_data']
    
    datasets = group_metadata(input_data.values(), 'dataset', sort=True)
    inp_date = datetime.strptime(cfg['analysis_day'], '%Y-%m-%d')

    for dataset in datasets.keys():
        metadata = datasets[dataset]
        filename = metadata[0]['filename']
        obs_cb = iris.load_cube(filename)
        last_day = obs_cb[-1]
        last_date = last_day.coord('time').cell(0).point
        if (inp_date.year==last_date.year)&(inp_date.month==last_date.month
                                            )&(inp_date.day==last_date.day):
            ref_cb = eprep.extract_time(obs_cb, 
                                        start_year=cfg['reference_period'][0],
                                        start_month=1, start_day=1,
                                        end_year=cfg['reference_period'][1],
                                        end_month=1, end_day=1)
            ref_cb = extract_partial_dates(ref_cb, inp_date,cfg['half_window'])
            ref_cb = eprep.annual_statistics(ref_cb, operator='max')
            ref_cb = eprep.climate_statistics(ref_cb)
            plot_cb = last_day - ref_cb
            plot_map(plot_cb, dataset, cfg)
            logger.info(f'Successfully processed {dataset}')
        else:
            logger.error('The input date '+cfg['analysis_day']+' from'
                         ' the recipe does not correspond to the last day'
                         ' in the observational cube')

    logger.info('Diagnostic completed successfully')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
