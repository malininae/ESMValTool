import iris
import iris.plot as iplt
import iris.coord_categorisation 
from iris.cube import Cube
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import geopandas as gpd 
import esmvalcore.preprocessor as eprep
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.extreme_events.map_distribution import obtain_cubes, define_inp_date

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def calculate_reference_cube(ref_cb: Cube, half_window: int, percentile: int):
    '''
    This function calculates climatological percentile in the provided window

    Input:
        ref_cb: the reference cube (time resolved) 
        half_window: the number of days around (+/-) the date of interest to 
                     consider to calculate the trigger percentile
        percentile: the trigger percentile to determine heatwave
    Output:
        clim_ref_cb: climatological cube with calculated percentiles
    '''

    # adding aux coord with the day of year 
    iris.coord_categorisation.add_day_of_year(ref_cb, 'time', 'doy')

    clim_perc = np.zeros(366)
    doy = ref_cb.coord('doy').points ; ref_data = ref_cb.data
    for day in range(1, 367):   
        good_plus = (doy - day) % 365 <= half_window 
        good_minus = (day - doy) % 365 <= half_window
        good = good_plus | good_minus
        perc = np.nanpercentile(ref_data[good], percentile)
        clim_perc[day-1] = perc

    doy_coord = iris.coords.DimCoord(np.arange(1, 367), long_name='day_of_year')

    clim_ref_cb = Cube(clim_perc, standard_name=ref_cb.standard_name, 
                       dim_coords_and_dims=[(doy_coord,0)],
                       long_name=ref_cb.long_name, var_name=ref_cb.var_name,
                       units=ref_cb.units, attributes=ref_cb.attributes)


    return clim_ref_cb   


def analyse_heatwave(obs_cb: Cube, ref_cb: Cube, inp_date: date):
    '''
    This function detects if there was a heatwave on the inp_day

    Input:
        obs_cb: iris cube from which the day is extracted
        ref_cb: the reference cube (time resolved) 
        inp_date: date around which needs to be extracted
    '''

    hw_len = 0; hw_idx = 0; hw_start = inp_date
    while hw_idx == hw_len:
        hw_idx += 1
        day_cb = obs_cb.extract(iris.Constraint(
                time=lambda cell: datetime.strptime(
                str(cell.point),'%Y-%m-%d %H:%M:%S').date()==hw_start))    
        day_of_y = inp_date.timetuple().tm_yday 
        ref_day_cb= ref_cb.extract(iris.Constraint(day_of_year=day_of_y))
        shape_id = day_cb.coord('shape_id').cell(0).point 
        if day_cb.data > ref_day_cb.data:
            logger.info(f'Warm day was detected on {hw_start} in {shape_id}')
            hw_start = hw_start - timedelta(days=1) 
            hw_len += 1
        else:
            logger.info(f'No warm day was detected on {hw_start} in {shape_id}')
            hw_start = hw_start + timedelta(days=1) 
            hw_end = inp_date
            break                

    if hw_start>inp_date:
        hw_start = None ; hw_end = None ; hw_max = None; hw_3max = None
    else: 
        if hw_len>3:
            hw_cb = obs_cb.extract(iris.Constraint(
                    time=lambda cell: hw_start <= datetime.strptime(
                    str(cell.point),'%Y-%m-%d %H:%M:%S').date() <= hw_end))
            hw_3cb = eprep.rolling_window_statistics(hw_cb, coordinate='time', 
                                             window_length=3, operator='mean')
            hw_max_cell = hw_cb.collapsed('time', iris.analysis.MAX).coord('time'
                                                                    ).cell(0).point
            hw_max = datetime.strptime(str(hw_max_cell),'%Y-%m-%d %H:%M:%S').date()
            hw_3max_cell = hw_3cb.collapsed('time', iris.analysis.MAX).coord('time'
                                                                    ).cell(0).point
            hw_3max = datetime.strptime(str(hw_3max_cell),'%Y-%m-%d %H:%M:%S').date()
        elif hw_len==1:
            hw_max = hw_start
            hw_3max = 'None'
        elif hw_len == 3:
            hw_cb = obs_cb.extract(iris.Constraint(
                    time=lambda cell: hw_start <= datetime.strptime(
                    str(cell.point),'%Y-%m-%d %H:%M:%S').date() <= hw_end))
            hw_3cb = eprep.rolling_window_statistics(hw_cb, coordinate='time', 
                                             window_length=3, operator='mean')
            hw_max_cell = hw_cb.collapsed('time', iris.analysis.MAX).coord('time'
                                                                    ).cell(0).point
            hw_max = datetime.strptime(str(hw_max_cell),'%Y-%m-%d %H:%M:%S').date()
            hw_3max_cell = hw_3cb.coord('time').cell(0).point
            hw_3max = datetime.strptime(str(hw_3max_cell),'%Y-%m-%d %H:%M:%S').date()
        else:
            hw_cb = obs_cb.extract(iris.Constraint(
                    time=lambda cell: hw_start <= datetime.strptime(
                    str(cell.point),'%Y-%m-%d %H:%M:%S').date() <= hw_end))
            hw_max_cell = hw_cb.collapsed('time', iris.analysis.MAX).coord('time'
                                                                    ).cell(0).point
            hw_max = datetime.strptime(str(hw_max_cell),'%Y-%m-%d %H:%M:%S').date()
            hw_3max = 'None'
 
    return hw_start, hw_end, hw_max, hw_3max, hw_len


def plot_heatwave_length(obs_cb: Cube, ref_cb: Cube, hw_info: dict, 
                                                      dataset: str, cfg: dict):
    '''
    This function creates a heatwave plot as Fig. 2 in  Malinina&Gillett (2024)

    Input:
        obs_cb: iris cube with year of analysis observations
        ref_cb: iris cube with reference period for observations
        hw_info: dictionary with dates and length of a heatwave
        dataset: name of the dataset
        cfg: standard ESMValTool config disctionary
    '''

    # define region name
    shape_id = obs_cb.coord('shape_id').cell(0).point

    hw_start = hw_info['hw_start'] ; hw_end = hw_info['hw_end']
    ext_st_prev = hw_start - relativedelta(month=1)
    ext_start = datetime(ext_st_prev.year, ext_st_prev.month, 1).date()
    ext_start_idx = ext_start.timetuple().tm_yday
    hw_end_idx = hw_end.timetuple().tm_yday

    perc_label = '('+str(cfg['trigger_percentile'])+' perc)'

    current_cb = obs_cb.extract(iris.Constraint(
                        time=lambda cell: ext_start <= datetime.strptime(
                        str(cell.point),'%Y-%m-%d %H:%M:%S').date() <= hw_end))
    ref_cb = ref_cb.extract(iris.Constraint(
                 day_of_year=lambda cell: ext_start_idx <= cell <= hw_end_idx))

    years = list(sorted(set([str(ext_start.year), str(hw_end.year)])))
    if len(years)>1:
        y_lbl = '('+years[0]+'-'+years[-1]+')'
    else:
        y_lbl = '('+years[0]+')'

    # loading matplotlib style saved in ESMValTool folder
    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    fig , ax = plt.subplots(1,1)
    
    fig.set_dpi(150)
    fig.set_size_inches(13,5)

    ax.plot(np.arange(current_cb.shape[0]), current_cb.data, 
                                 label=cfg['var_label'] + y_lbl, c='darkred')
    ax.plot(np.arange(ref_cb.shape[0]), ref_cb.data, label=cfg['var_label']+ \
                        '$_{'+perc_label+'}$' + '(' + \
                        str(cfg['reference_period'][0]) + '-' + \
                        str(cfg['reference_period'][1])+')', c='darkgrey')
    end_fpoint = ref_cb.shape[0]
    start_fpoint = end_fpoint - hw_info['hw_len'] - 1
    ax.fill_between(np.arange(start_fpoint, end_fpoint), 
                    current_cb.data[start_fpoint: end_fpoint], 
                    ref_cb.data[start_fpoint: end_fpoint],
                    where=current_cb.data[start_fpoint:end_fpoint]>ref_cb.data[start_fpoint:end_fpoint],
                    color='darkred', alpha=0.3, interpolate=True)
    ax.legend(loc = 0, fancybox=False, frameon=False, fontsize='large')
    ax.grid(color='silver', axis='both', alpha=0.5)
    ax.set_ylabel(cfg['var_label'] +' ,'+cfg['var_units'])
    ax.set_xlabel('date (month/day)')
    ax.set_xlim(0,end_fpoint-1)
    ax.text(0.85, 0.15,'Heatwave length: ' +str(hw_info['hw_len'])+' days\n('+ \
                                    str(hw_start)[5:].replace('-', '/')+' - '+\
                                    str(hw_end)[5:].replace('-', '/')+')',transform=ax.transAxes)
    xlabels = [str(d)[5:].replace('-','/') for d in np.arange(ext_start, hw_end+timedelta(days=1), 7)]
    ax.set_xticks(np.arange(0, end_fpoint, 7), labels=xlabels)
    fig.suptitle(f'{dataset} temperature in {shape_id} in '+y_lbl[1:-1], fontsize='xx-large')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg['plot_dir'], 
                            f'hw_{shape_id}_{dataset}.'+cfg['output_file_type']))


    return


def main(cfg):
    '''
    This function does data processing and initiates analysis and plotting.

    Input: 
        cfg: standard ESMValTool config disctionary
    '''

    last_day_l = cfg.get('last_day'); inp_day = cfg.get('analysis_day')
    if last_day_l== inp_day == None:
        logger.error('Either last_day or analysis_day should be provided in '
                                                                  'the recipe')

    input_data = cfg['input_data']
    
    datasets = group_metadata(input_data.values(), 'dataset', sort=True)

    for dataset in datasets.keys():

        current_cb, ref_cb = obtain_cubes(datasets[dataset], cfg)

        inp_date = define_inp_date(current_cb, last_day_l, inp_day)

        # creating a csv file for writing out regional heatwave info
        dataset_csv = open(os.path.join(cfg['work_dir'], 
                     f'{dataset}_regional_heatwave_info.csv'), 'w', newline='')
        dataset_csv_w = csv.writer(dataset_csv, delimiter=',')
        dataset_csv_w.writerow(['Region', 'start_day', 'end_day',
                                                       'max_day', '3_day_max', 'length'])

        for shape_id in current_cb.coord('shape_id').points:
            reg_obs_cb = current_cb.extract(iris.Constraint(shape_id=shape_id))
            reg_ref_cb = ref_cb.extract(iris.Constraint(shape_id=shape_id))
            reg_ref_cb = calculate_reference_cube(reg_ref_cb, 
                                                  cfg['half_window'],
                                                  cfg['trigger_percentile'])
            hw_start, hw_end, hw_max, hw_3max, hw_len = analyse_heatwave(reg_obs_cb,
                                                          reg_ref_cb, inp_date)
            if hw_start is not None:
                dataset_csv_w.writerow([shape_id, hw_start, 
                                        hw_end, hw_max, hw_3max, hw_len])
                hw_info = {'hw_start': hw_start, 'hw_end': hw_end,
                        'hw_max': hw_max, 'hw_3max': hw_3max, 'hw_len': hw_len}
                plot_heatwave_length(reg_obs_cb, reg_ref_cb, hw_info, dataset, cfg)
        
        dataset_csv.close()

        logger.info(f'Successfully processed {dataset}')
    
    logger.info('Diagnostic completed successfully')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
