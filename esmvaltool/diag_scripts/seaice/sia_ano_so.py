import iris
import esmvalcore.preprocessor as eprep
import cftime
import cf_units
import datetime
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
    try:
        cube.remove_coord('area_type')
    except:
        pass
    aux_coords = cube.aux_coords

    if cube.data.max()>1:
        siconc = cube.data / 100 # since the data is in %, it has to be converted into fraction
    else:
        siconc = cube.data
    area = np.ma.array(area, mask=cube.data.mask)

    sia_arr = siconc * area

    sia = (sia_arr.sum(axis=(1, 2)) / (1000 ** 2)) / 1000000 # iris provides area in m, converting it to 10^6km2
    # for now passing parent cube attributes, clean before merging!!!
    sia_cube = iris.cube.Cube(sia, standard_name='sea_ice_area', long_name='sea ice area', var_name='siarea',
                            aux_coords_and_dims = [(aux_coords[0],0), (aux_coords[1],0)], # think of better ways
                            units="1e6 km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])


    return (sia_cube)

def calculate_n_y_stats(data, proj_time_range, n_y=10, weights=None):

    if weights is None:
        if len(data.shape) == 1:
            ny_std = [np.std(data[n_y*i:n_y*i+n_y]) for i in range(0, data.shape[0]//n_y)]
        elif len(data.shape) == 2:
            ny_std = [np.std(data[:,n_y*i:n_y*i+n_y]) for i in range(0, data.shape[1]//n_y)]
    else:
        mmm = [np.average(data[:,n_y*i:n_y*i+n_y], weights=weights[:,n_y*i:n_y*i+n_y]) for i in range(0, data.shape[1]//n_y)]
        ny_std = [np.sqrt(np.average((data[:,n_y*i:n_y*i+n_y] - mmm[i])**2, weights=weights[:,n_y*i:n_y*i+n_y])) for i in range(0, data.shape[1]//n_y)]

    ny_std = np.asarray(ny_std).reshape(1,len(ny_std))

    dttm_dates = [datetime.datetime.strptime(str(y)+'-01-15', '%Y-%m-%d') for y in np.arange(proj_time_range[0]+n_y//2, proj_time_range[1], n_y)] 
    num_dates = cftime.date2num(dttm_dates, 'days since 1850-01-01', calendar='gregorian')
    tim_coord = iris.coords.DimCoord(num_dates, standard_name = 'time', long_name='time', var_name= 'time',
                                    units = cf_units.Unit('days since 1850-01-01', calendar='gregorian'))

    num_coord = iris.coords.DimCoord([1])

    ny_std_cb = iris.cube.Cube(ny_std, long_name = 'siarea, '+ str(n_y)+'y std', var_name = 'siarea',
                                units = "1e6 km2", dim_coords_and_dims=[(tim_coord, 1)]) #, (num_coord, 0)])

    return ny_std_cb


def calculate_multi_stats(proj_dic, project, proj_time_range, n_y=10):

    out_dic = {}

    if project != 'OBS': 
        out_dic['Multi-Model'] = {} 
        exps = []
        dtsts = []
        for mod in proj_dic.keys():
            dtsts.append(mod)
            out_dic[mod] = {}
            for exp in proj_dic[mod].keys():
                out_dic[mod][exp] = {}
                exps.append(exp)
        exps = set(exps)
        for exp in exps:
            model_data = []
            weights = []
            for dtst in dtsts:
                exp_dtst_cont = proj_dic[dtst][exp]
                mod_time = exp_dtst_cont[0].coord('time')
                if len(exp_dtst_cont) >1:
                    data = np.zeros((len(exp_dtst_cont), len(mod_time.points)))
                    for nm, mod_cb in enumerate(exp_dtst_cont):
                        data[nm,:] = mod_cb.data
                        model_data.append(mod_cb.data)
                        weights.append(np.full(len(mod_time.points), 1/len(exp_dtst_cont)))
                    mod_mean = np.mean(data, axis=0)
                    n_y_std_cb = calculate_n_y_stats(data, proj_time_range, n_y= n_y)
                    mod_mean_cb = iris.cube.Cube(mod_mean, standard_name = exp_dtst_cont[0].standard_name, 
                                                long_name=exp_dtst_cont[0].long_name, var_name=exp_dtst_cont[0].var_name, 
                                                units=exp_dtst_cont[0].units, dim_coords_and_dims=[(mod_time, 0)])
                    out_dic[dtst][exp][str(n_y)+'_y_std'] = n_y_std_cb
                    out_dic[dtst][exp]['mean_ts'] = mod_mean_cb
                    out_dic[dtst][exp]['individual_ts'] = exp_dtst_cont
                else:
                    out_dic[dtst][exp]['mean_ts'] = exp_dtst_cont[0]
                    n_y_std_cb = calculate_n_y_stats(exp_dtst_cont[0].data, proj_time_range, n_y= n_y)
                    out_dic[dtst][exp][str(n_y)+'_y_std'] = n_y_std_cb
                    model_data.append(exp_dtst_cont[0].data)
                    weights.append(np.full(len(exp_dtst_cont[0].coord('time').points), 1))

            model_data = np.asarray(model_data)
            weights = np.asarray(weights) / len(dtsts)

            out_dic['Multi-Model'][exp] = {}
            
            out_dic['Multi-Model'][exp][str(n_y)+'_y_std'] = calculate_n_y_stats(model_data, proj_time_range, n_y=n_y, weights=weights)

            mult_m_mean = np.average(model_data, axis=0, weights=weights)
 
            out_dic['Multi-Model'][exp]['mean_ts'] = mult_m_mean
            out_dic['Multi-Model'][exp]['std_ts'] = np.sqrt(np.average((model_data - mult_m_mean)**2, axis=0, weights=weights))
    else:
        for dtst in proj_dic.keys():
            out_dic[dtst] = {'ts': proj_dic[dtst]}
            out_dic[dtst][str(n_y)+'_y_std'] = calculate_n_y_stats(proj_dic[dtst].data, proj_time_range, n_y=n_y)

    return out_dic    

def plot_timeseries(data_dic, cfg):

    projs = list(data_dic.keys()); projs.remove('OBS')

    cols = {'historical': (0.37890625, 0.32421875, 0.796875),
            'ssp245' : (0.6484375, 0., 0.40234375),
            'piControl': (0.61328125, 0.79296875, 0.7265625) }

    for proj in projs:
        models = list(data_dic[proj].keys())
        for model in models:
            fig_ts, ax_ts = plt.subplots(1)
            fig_ts.set_size_inches(12.,8.)
            for exp in data_dic[proj][model].keys(): 
                if model != 'Multi-Model': 
                    to_plot_data = data_dic[proj][model][exp]['mean_ts'].data
                else: 
                    to_plot_data = data_dic[proj][model][exp]['mean_ts']
                if exp == 'piControl':
                    ax_ts.plot(np.arange(cfg[proj.lower()+'_time_range'][0], cfg[proj.lower()+'_time_range'][1]+1), 
                           to_plot_data, zorder = 100 , c = cols['piControl'], lw = 2.5, label = model + ' piControl mean')
                else: 
                    ax_ts.plot(np.arange(cfg[proj.lower()+'_time_range'][0], 2015), 
                           to_plot_data[:30], zorder = 100 , c = cols['historical'], lw = 2.5, label = model + ' historical mean')
                    ax_ts.plot(np.arange(2015, cfg[proj.lower()+'_time_range'][1]+1), 
                           to_plot_data[30:], zorder = 100 , c = cols['ssp245'], lw = 2.5, label = model + ' ssp245 mean')
                if data_dic[proj][model][exp].get('individual_ts') is not None:
                    n_rs = len(data_dic[proj][model][exp]['individual_ts'])
                    for n_r in range(n_rs):
                        if exp == 'piControl':
                            ax_ts.plot(np.arange(cfg[proj.lower()+'_time_range'][0], cfg[proj.lower()+'_time_range'][1]+1),
                                    data_dic[proj][model][exp]['individual_ts'][n_r].data, 
                                    zorder = 80-n_r , c = cols['piControl'], lw = 0.5, alpha=0.5)
                        else: 
                            ax_ts.plot(np.arange(cfg[proj.lower()+'_time_range'][0], 2015), 
                                        data_dic[proj][model][exp]['individual_ts'][n_r].data[:30], 
                                        zorder = 100 , c = cols['historical'], lw = 0.5, alpha=0.5)
                            ax_ts.plot(np.arange(2015, cfg[proj.lower()+'_time_range'][1]+1), 
                                        data_dic[proj][model][exp]['individual_ts'][n_r].data[30:], 
                                        zorder = 100 , c = cols['ssp245'], lw = 0.5, alpha=0.5)
                elif data_dic[proj][model][exp].get('std_ts') is not None:
                    if exp == 'piControl':
                        ax_ts.fill_between(np.arange(cfg[proj.lower()+'_time_range'][0], cfg[proj.lower()+'_time_range'][1]+1), 
                                        to_plot_data + data_dic[proj][model][exp]['std_ts'],
                                        to_plot_data - data_dic[proj][model][exp]['std_ts'], zorder = 99 , 
                                        color = cols['piControl'], linewidth = 0, alpha=0.3)
                    else:
                        ax_ts.fill_between(np.arange(cfg[proj.lower()+'_time_range'][0], 2015), 
                                        to_plot_data[:30] + data_dic[proj][model][exp]['std_ts'][:30],
                                        to_plot_data[:30] - data_dic[proj][model][exp]['std_ts'][:30], zorder = 99 , 
                                        color = cols['historical'], linewidth = 0, alpha=0.2)
                        ax_ts.fill_between(np.arange(2015, cfg[proj.lower()+'_time_range'][1]+1), 
                                        to_plot_data[30:] + data_dic[proj][model][exp]['std_ts'][30:],
                                        to_plot_data[30:] - data_dic[proj][model][exp]['std_ts'][30:], zorder = 99 , 
                                        color = cols['ssp245'], linewidth = 0, alpha=0.2)
            for n_o, obs_dtst in enumerate(list(data_dic['OBS'].keys())):
                cnum = list(np.full(3, 10 + n_o*190/len(list(data_dic['OBS'].keys())))/255)
                plt.plot(np.arange(cfg['obs_time_range'][0], cfg['obs_time_range'][1]+1), 
                                    data_dic['OBS'][obs_dtst]['ts'].data, c=cnum, lw=1.5, label= obs_dtst, zorder=98-n_o)
            ax_ts.hlines(0, cfg['cmip6_time_range'][0]-1, cfg['cmip6_time_range'][1]+2, zorder=1, color='silver', linestyle='dashed', linewidth = 1)
            ax_ts.set_xlim(cfg['cmip6_time_range'][0]-1, cfg['cmip6_time_range'][1]+2)
            ax_ts.set_xlabel('year')
            ax_ts.set_ylabel('SIAa, 1e6 km2')
            ax_ts.legend(loc=0, ncols=2, fancybox=False, frameon=False)
            ax_ts.set_title('DJF time series of Sea Ice Area Anomalies (SIAa) from '+ model)
            plt.tight_layout()
            fig_ts.savefig(os.path.join(cfg['plot_dir'], 'timeseries_'+ proj +'_'+ model + diagtools.get_image_format(cfg)))

    return

def plot_stds(data_dic, cfg):

    ny= cfg['n_year_std']

    projs = list(data_dic.keys()); projs.remove('OBS')
    for proj in projs:
        models = list(data_dic[proj].keys())
        for model in models:
            fig_ts, ax_ts = plt.subplots(len(data_dic[proj][model].keys()) +len(data_dic['OBS'].keys()), 1, sharex=True)
            ax_ts = ax_ts.flatten()
            fig_ts.set_size_inches(12.,8.)
            for nobsx, obs in enumerate(sorted(data_dic['OBS'].keys())):
                obs_std_cb= data_dic['OBS'][obs][str(ny)+'_y_std']
                pmesh = iplt.pcolormesh(obs_std_cb, axes=ax_ts[nobsx],
                                         cmap=plt.cm.Reds, vmax=1, vmin=0)
                ax_ts[nobsx].set_title(obs +' (OBS)')   
            for nax, exp in enumerate(sorted(data_dic[proj][model].keys())):
                std_cb = data_dic[proj][model][exp][str(ny)+'_y_std']
                iplt.pcolormesh(std_cb, axes=ax_ts[len(data_dic['OBS'].keys())+nax], cmap=plt.cm.Reds, vmax=1, vmin=0)
                ax_ts[len(data_dic['OBS'].keys())+nax].set_title(exp)                   
            xlim = ax_ts[-1].get_xlim()  
            step = (xlim[1] -xlim[0])/11
            ax_ts[-1].set_xticks(np.arange(xlim[0]+0.5*step, xlim[1], step))
            ax_ts[-1].set_xticklabels([str(y)+'-\n'+str(y+9) for y in np.arange(1985, 2090, 10)])
            [a.spines['bottom'].set_visible(False) for a in ax_ts]
            [a.spines['left'].set_visible(False) for a in ax_ts]
            [a.tick_params(axis='x', length=0) for a in ax_ts[:-1]]
            [a.get_yaxis().set_visible(False) for a in ax_ts]
            plt.tight_layout()
            cax = fig_ts.add_axes([0.2,0.075,0.6,0.025])
            cbar = fig_ts.colorbar(pmesh, cax=cax, orientation='horizontal')
            cbar.ax.set_xlabel(r'10$^6$ km$^2$' +' SIAa standard deviation')
            fig_ts.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.17, wspace=0.23)  # , hspace=0.2)
            fig_ts.suptitle('Standard deviation of '+str(ny)+' year DJF Sea Ice Area Anomalies (SIAa)', fontsize='x-large')
            fig_ts.savefig(os.path.join(cfg['plot_dir'], 'std_'+ proj +'_'+ model + diagtools.get_image_format(cfg)))

    return   

def main(cfg):
    
    input_data = cfg['input_data']

    projects = group_metadata(input_data.values(), 'project', sort=True)
    projects_l = list(projects.keys())

    sia_dic = {}

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    sia_dic = {}
    for project in projects_l:
        sia_dic[project] = {}
        proj_info = projects[project]
        datasets = group_metadata(proj_info, 'dataset')
        proj_dic = {}
        for dataset in datasets.keys():
            proj_dic[dataset] = {}
            if project != 'OBS':
                exps = group_metadata(datasets[dataset], 'exp')
                for exp in exps.keys():
                    filepaths = list(group_metadata(exps[exp], 'filename').keys())
                    mod_cubes = iris.cube.CubeList()
                    for filepath in filepaths: 
                        mod_cb = iris.load_cube(filepath)
                        sia_cb = calculate_siarea(mod_cb)
                        sia_seas_cb = eprep.seasonal_statistics(sia_cb, operator='mean', seasons=['DJF'])
                        # determine the first year of the dataset
                        start_year = input_data[filepath]['start_year']
                        end_ref_year = start_year + cfg['ref_period_len']
                        # including the 1st of January next year depending on how time stamp falls 
                        sia_ano_cb = eprep.anomalies(sia_seas_cb, 'full',
                            reference={'start_year': start_year,
                            'start_month': 1, 'start_day':1,
                            'end_year': end_ref_year,
                            'end_month': 1, 'end_day':1})
                        mod_cubes.append(sia_ano_cb)
                    proj_dic[dataset][exp] = mod_cubes
            else: 
                filepath = list(group_metadata(datasets[dataset], 'filename').keys())[0]
                obs_cb = iris.load_cube(filepath)
                # that'a temporary fix. We remove 1987-88 because 2 months out of 3 didn't have data
                # so far it's done through mask assignment of the 4th year. TODO: improve
                if obs_cb.data.mask == False:
                    obs_cb.data.mask = np.zeros(len(obs_cb.data), dtype=bool)
                obs_cb.data.mask[3] = True
                obs_cb.data.data[3] = obs_cb.data.fill_value
                start_year= input_data[filepath]['start_year']
                end_ref_year = start_year + cfg['ref_period_len']
                obs_ano_cb = eprep.anomalies(obs_cb, 'full',
                            reference={'start_year': start_year,
                            'start_month': 1, 'start_day':1,
                            'end_year': end_ref_year,
                            'end_month': 1, 'end_day':1})
                proj_dic[dataset] = obs_ano_cb
        sia_dic[project] = calculate_multi_stats(proj_dic, project, cfg[project.lower()+'_time_range'], 
                                                                                n_y = cfg['n_year_std'])
    plot_timeseries(sia_dic, cfg)

    plot_stds(sia_dic, cfg)
    
    # plot meshplot for obs and CMIP6 

    logger.info('Succes')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)