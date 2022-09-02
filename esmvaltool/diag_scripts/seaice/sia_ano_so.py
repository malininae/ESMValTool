from turtle import color
import iris
import esmvalcore.preprocessor as eprep
from iris.util import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def calculate_siarea(cube):

    # calculates siarea for the hemisphere
    # creates a cube with only one dimension: 'time'

    area = iris.analysis.cartography.area_weights(cube, normalize=False)
    time = cube.coord('time')

    siconc = cube.data / 100 # since the data is in %, it has to be converted into fraction
    area = np.ma.array(area, mask=cube.data.mask)

    sia_arr = siconc * area

    sia = (sia_arr.sum(axis=(1, 2)) / (1000 ** 2)) / 1000000 # iris provides area in m, converting it to 10^6km2
    # for now passing parent cube attributes, clean before merging!!!
    sia_cube = iris.cube.Cube(sia, standard_name='sea_ice_area', long_name='sea ice area', var_name='siarea',
                            units="1e6 km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])


    return (sia_cube)

def calculate_multi_stats(proj_dic):

    out_dic = {}

    model_data = list()
    weights = list()

    for model in proj_dic.keys():
        mod_cont = proj_dic[model]
        out_dic[model] = {}
        if type(mod_cont) is iris.cube.CubeList: 
            mod_time = mod_cont[0].coord('time')
            data = np.zeros((len(mod_time.points),len(mod_cont)))
            for nm, mod_cb in enumerate(mod_cont):
                data[:,nm] = mod_cb.data
                if type(model_data) is list:
                    model_data.append(mod_cb.data)
                    weights.append(np.full(len(mod_time.points), 1/len(mod_cont)))
                else: 
                    model_data = np.vstack([model_data, mod_cb.data])
                    weights = np.vstack([weights, np.full(len(mod_time.points), 1/len(mod_cont))])
            mod_mean = np.mean(data, axis=1)
            mod_mean_cb = iris.cube.Cube(mod_mean, standard_name = mod_cont[0].standard_name, 
                                        long_name=mod_cont[0].long_name, var_name=mod_cont[0].var_name, 
                                        units=mod_cont[0].units, dim_coords_and_dims=[(mod_time, 0)])
            out_dic[model]['mean'] = mod_mean_cb
            out_dic[model]['individual'] = mod_cont
        elif type(mod_cont) is iris.cube.Cube:
            out_dic[model]['mean'] = mod_cont
            if type(model_data) is list:
                model_data.append(mod_cont.data)
                weights.append(np.full(len(mod_cont.coord('time').points), 1))
            else: 
                model_data = np.vstack([model_data, mod_cont.data])
                weights = np.vstack([weights, np.full(len(mod_cont.coord('time').points), 1)])

    model_data = np.asarray(model_data)
    weights = np.asarray(weights)

    mult_m_mean = np.average(model_data, axis=0, weights=weights)

    out_dic['Multi-Model'] = {} 
    out_dic['Multi-Model']['mean'] = mult_m_mean
    out_dic['Multi-Model']['std'] = np.sqrt(np.average((model_data - mult_m_mean)**2, axis=0, weights=weights))

    return out_dic    

def plot_timeseries(data_dic, cfg, month):

    projs = list(data_dic.keys()); projs.remove('OBS')

    for proj in projs:
        models = list(data_dic[proj].keys())
        for model in models:
            fig_ts, ax_ts = plt.subplots(1)
            fig_ts.set_size_inches(12.,8.)
            n_reals = ''
            if model != 'Multi-Model':
                ax_ts.plot(np.arange(1979,2021), data_dic[proj][model]['mean'].data, zorder = 100 , c = 'tab:blue', lw = 2.5, label = model + ' mean')
                try: 
                    n_rs = len(data_dic[proj][model]['individual'])
                    for n_r in range(n_rs):
                        ax_ts.plot(np.arange(1979,2021) ,data_dic[proj][model]['individual'][n_r].data, zorder = 80-n_r , c = 'tab:blue', lw = 0.5, alpha=0.5)
                        n_reals = ' N='+str(n_rs)
                except: 
                    pass
            else:
                ax_ts.plot(np.arange(1979,2021), data_dic[proj][model]['mean'], zorder = 100 , c = 'tab:blue', lw = 2.5, label = model + ' mean')
                ax_ts.fill_between(np.arange(1979,2021), data_dic[proj][model]['mean'] + data_dic[proj][model]['std'],
                                data_dic[proj][model]['mean']-data_dic[proj][model]['std'], zorder = 99 , color = 'tab:blue', linewidth = 0, alpha=0.3)
            for n_o, obs_dtst in enumerate(list(data_dic['OBS'].keys())):
                cnum = list(np.full(3, 10 + n_o*190/len(list(data_dic['OBS'].keys())))/255)
                plt.plot(np.arange(1979, 2020), data_dic['OBS'][obs_dtst].data, c=cnum, lw=1.5, label= obs_dtst, zorder=98-n_o)
            ax_ts.hlines(0, 1978,2022, zorder=1, color='silver', linestyle='dashed', linewidth = 1)
            ax_ts.set_xlim(1978, 2021)
            ax_ts.set_xlabel('year')
            ax_ts.set_ylabel('SIAa, 1e6 km2')
            ax_ts.legend(loc=0, fancybox=False, frameon=False)
            ax_ts.set_title('Time series of Sea Ice Area Anomalies (SIAa) in ' + str(month) +' from '+ model + n_reals)
            plt.tight_layout()
            fig_ts.savefig(os.path.join(cfg['plot_dir'], 'timeseries_'+ proj +'_'+ model + '_' + str(month) + diagtools.get_image_format(cfg)))

    return

def main(cfg):
    
    input_data = cfg['input_data']

    projects = group_metadata(input_data.values(), 'project', sort=True)
    projects_l = list(projects.keys())

    months = cfg['months']

    sia_dic = {}

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    for month in months:
        sia_dic[month] = {}
        for project in projects_l:
            sia_dic[month][project] = {}
            proj_info = projects[project]
            datasets = group_metadata(proj_info, 'dataset')
            proj_dic = {}
            for dataset in datasets.keys():
                filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
                n_real = len(filepaths)
                mod_cubes = iris.cube.CubeList()
                for filepath in filepaths: 
                    mod_cb = iris.load_cube(filepath)
                    month_mod_cb = eprep.extract_month(mod_cb, month)
                    if project != 'OBS': 
                        sia_cb = calculate_siarea(month_mod_cb)
                        sia_ano_cb = eprep.anomalies(sia_cb, 'monthly',
                            reference={'start_year': cfg['ref_period'][0],
                            'start_month': 1, 'start_day':1,
                            'end_year': cfg['ref_period'][1],
                            'end_month': 12, 'end_day':31})
                        sia_ano_cb = eprep.annual_statistics(sia_ano_cb, 'mean') # needed to get the same timestamp
                    else: 
                        sia_ano_cb = eprep.annual_statistics(month_mod_cb, 'mean') # needed to get the same timestamp
                    if n_real>1:
                        mod_cubes.append(sia_ano_cb)
                    else:
                        mod_cubes = sia_ano_cb
                proj_dic[dataset] = mod_cubes
            if project != 'OBS':
                sia_dic[month][project] = calculate_multi_stats(proj_dic)
            else: 
                sia_dic[month][project] = proj_dic
        plot_timeseries(sia_dic[month], cfg, month)
    
    # plot meshplot for obs and CMIP6 

    logger.info('Succes')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)