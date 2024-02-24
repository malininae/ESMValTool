import iris
import iris.plot as iplt
from iris.cube import Cube
from iris.time import PartialDateTime
import cartopy.crs as ccrs 
import geopandas as gpd 
import esmvalcore.preprocessor as eprep
from datetime import datetime, timedelta, date
import logging
import numpy as np
import matplotlib.pyplot as plt
import os


# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def obtain_cubes(dataset: list, cfg: dict):
    '''
    This function retrieves current and reference cube from datasets.

    Input: 
        datasets: list with dictionaries based on ESMValTool input_data
    Output:
        current_cube: iris Cube with current data
        ref_cube: iris Cube with reference data
    '''

    current_f = select_metadata(dataset,variable_group='current'
                                                        )[0]['filename']
    diagnostic = select_metadata(dataset,variable_group='current'
                                                        )[0]['diagnostic']
    try:
        # checking if the cube exist in aux directory
        current_base = os.path.basename(current_f)
        f_type = current_base.split('.')[-1]
        # 10 comes from '.' + 4 for start_year + '-' + 4 for end_year
        # TODO come with better solution
        ref_base = current_base[:len(f_type)+10] + \
                                str(cfg['reference_period'][0]) + '-' + \
                                str(cfg['reference_period'][1]+'.'+ f_type)
        ref_f = os.path.join(cfg['auxiliary_data_dir'], diagnostic, ref_base)
    except:
        ref_f = select_metadata(dataset,variable_group='reference'
                                                        )[0]['filename']
    logger.info(f'Found file {ref_f}')

    current_cb = iris.load_cube(current_f)
    ref_cb = iris.load_cube(ref_f)

    return current_cb, ref_cb


def define_inp_date(cube: Cube, last_day: bool | None, inp_day: str | None):
    '''
    This function defines explicitly the inp_date

    Input:
        cube: iris Cube with the data
        last_day: boolean or None, defines if we should use the last day 
        inp_day: str or None, date which will be used for analysis
    Output: 
        inp_date: date which will be used for analysis
    '''

    if inp_day!=None:
        inp_date = datetime.strptime(inp_day, '%Y-%m-%d').date()
    
    if last_day:
        day_cb = cube[-1]
        day_date = datetime.strptime(
            str(day_cb.coord('time').cell(0).point),'%Y-%m-%d %H:%M:%S').date()
        if (inp_day!=None)&((inp_date.year!=day_date.year
                                )|(inp_date.month!=day_date.month
                                        )|(inp_date.day!=day_date.day)):
            logger.warning(f'The input date {inp_day} from the recipe does '
                   'not correspond to the last day in the observational cube '
                   f'{day_date}. Proceeding with the last day.')
        inp_date = datetime.strptime(
            str(day_cb.coord('time').cell(0).point),'%Y-%m-%d %H:%M:%S').date()

    return inp_date


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
    fig.set_size_inches(7., 9.)
    fig.set_dpi(200)

    max_t = cube.collapsed(['latitude', 'longitude'], iris.analysis.MAX).data
    min_t = cube.collapsed(['latitude', 'longitude'], iris.analysis.MIN).data
    bord = np.floor(np.abs(np.asarray([max_t, min_t]).flatten()).max()*0.95)
    ax.coastlines(linewidth=0.5)
    map_pl = iplt.contourf(cube, axes=ax, levels=np.arange(-bord, bord+1,1), 
                                                cmap='RdBu_r', extend='both')
    ax.set_title('Maximum '+cfg['var_label'].lower()+' anomalies in ' + \
              cfg['region'] + ' (' +str(datetime.strptime(str(
              cube.coord('time').cell(0).point), '%Y-%m-%d %H:%M:%S').date())+\
              ')', fontsize='x-large')
    shape_gpd.boundary.plot(color=None, edgecolor='grey', linewidth=0.5, ax=ax, 
                                                transform=ccrs.PlateCarree())
    
    ax.set_extent([-60.5, -120.5,37.8,82.8])  
    ax.set_position([0.021, 0.315, 0.957, 0.64])

    cax = fig.add_axes([0.117, 0.28, 0.78, 0.015])
    fig.colorbar(map_pl, cax=cax, orientation='horizontal',
                                              label='Temperature anomalies, C')

    n_cols = 4 ; n_rows= int(np.ceil(len(shape_gpd)/n_cols))

    for n in range(len(shape_gpd)):
        ax.text(list(shape_gpd.iloc[n].geometry.centroid.coords)[0][0],
                list(shape_gpd.iloc[n].geometry.centroid.coords)[0][1],
                str(n+1), fontsize='x-large',  transform=ccrs.PlateCarree())
        fig.text(0.021+(n//n_rows)*0.957/n_cols, 0.2-(n%n_rows)*0.19/n_rows, 
                 str(n+1)+'. '+shape_gpd.iloc[n].ID)
        

    # add the legend for the regions  
    fig.savefig(os.path.join(cfg['plot_dir'], 
                                f'map_canada_{dataset}.'+cfg['output_file_type']))

    return

def main(cfg):

    last_day_l = cfg.get('last_day'); inp_day = cfg.get('analysis_day')
    if last_day_l== inp_day == None:
        logger.error('Either last_day or analysis_day should be provided in '
                                                                  'the recipe')

    input_data = cfg['input_data']
    
    datasets = group_metadata(input_data.values(), 'dataset', sort=True)

    for dataset in datasets.keys():
        
        current_cb, ref_cb = obtain_cubes(datasets[dataset], cfg)

        inp_date = define_inp_date(current_cb, last_day_l, inp_day)

        day_cb = current_cb.extract(iris.Constraint(
                    time=lambda cell: datetime.strptime(
                    str(cell.point),'%Y-%m-%d %H:%M:%S').date()==inp_date))

        day_of_y = inp_date.timetuple().tm_yday
        ref_day_cb= ref_cb.extract(iris.Constraint(day_of_year=day_of_y))
        
        plot_cb = day_cb - ref_day_cb
        plot_map(plot_cb, dataset, cfg)
        logger.info(f'Successfully processed {dataset}')

    logger.info('Diagnostic completed successfully')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
