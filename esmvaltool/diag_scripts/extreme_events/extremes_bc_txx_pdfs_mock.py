import cf_units
import cftime
import datetime
import csv

import esmvalcore.preprocessor
import iris
from iris.util import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import regrid
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def anomaly_rxNday(cfg):

    anomaly_metadata = select_metadata(cfg['input_data'].values(), variable_group='anomaly')

    new_fdir = os.path.join(cfg['work_dir'], 'anomaly')
    if not os.path.exists(new_fdir):    
        os.makedirs(new_fdir)

    for single_anomaly in anomaly_metadata: 
        orig_fname = single_anomaly['filename']
        f_name = orig_fname.split('/')[-1]
        new_fname = os.path.join(new_fdir, 'rx'+str(cfg['n_days'])+'day_'+f_name)
        ano_cb = iris.load_cube(orig_fname)
        rxNd_cube = ano_cb.rolling_window('time', iris.analysis.SUM, cfg['n_days'])
        if (cfg['mult_factor'] != 1):
            rxNd_cube = rxNd_cube * cfg['mult_factor']
        rxNd_cube = esmvalcore.preprocessor.annual_statistics(rxNd_cube, operator='max')
        rxNd_cube = esmvalcore.preprocessor.climate_statistics(rxNd_cube,
                                                               operator='mean',
                                                               period='full')
        iris.save(rxNd_cube, os.path.join(new_fdir,new_fname))
        cfg['input_data'][new_fname] = cfg['input_data'].pop(orig_fname)
        cfg['input_data'][new_fname]['filename'] = new_fname

    return


def get_era_txx(cfg):

    aux_dir = cfg['auxiliary_data_dir']
    work_dir = cfg['work_dir']
    pattern = cfg['era_fname_pattern']
    cont_aux = sorted(glob.glob(aux_dir + '/era5_' + pattern+'*.nc'))

    era_dir = os.path.join(work_dir, 'era5')
    if not os.path.exists(era_dir):
        os.makedirs(era_dir)

    for era_fname in cont_aux: 
        f_ending = era_fname[len(aux_dir)+6+len(pattern):] 
        era_cb = iris.load_cube(era_fname) 
        if pattern == 'total_precipitation':
            era_cb = esmvalcore.preprocessor.daily_statistics(era_cb, operator = 'sum')*1000
        regrid_cube = esmvalcore.preprocessor.regrid(era_cb, {'start_longitude' : 217.25, 
                                                       'end_longitude' : 310.25, 
                                                       'step_longitude' : 0.5,
                                                       'start_latitude' : 39.5, 
                                                       'end_latitude' : 84.5,
                                                       'step_latitude' : 0.5}, 'linear') 
        reg_cube = esmvalcore.preprocessor.extract_region(regrid_cube, cfg['era_regridding_region'][0], cfg['era_regridding_region'][1], 
                        cfg['era_regridding_region'][2], cfg['era_regridding_region'][3])
        era_cb_pr = esmvalcore.preprocessor.area_statistics(reg_cube, 'mean')
        iris.save(era_cb_pr, os.path.join(era_dir,'daily_era5_'+pattern +'_'+f_ending)) 

    era_files = sorted(glob.glob(era_dir + '/daily_era5_'+pattern +'*'))

    era_cubelist = iris.load(era_files)
    equalise_attributes(era_cubelist)

    for n in range(len(era_cubelist)): 
        era_cubelist[n].data = era_cubelist[n].data.astype('float32')
        if era_cubelist[n].coord('time').units != era_cubelist[0].coord('time').units: 
            era_cubelist[n].coord('time').units = era_cubelist[0].coord('time').units

    era_big_cube = era_cubelist.concatenate_cube()

    era_rxNday_big_cube = era_big_cube.rolling_window('time', iris.analysis.SUM, cfg['n_days'])

    era_rxNday_big_cube = esmvalcore.preprocessor.annual_statistics(era_rxNday_big_cube, 'max')  
    
    # if cfg['era_regridding_shape']: 
    #     reg_cb_txx = esmvalcore.preprocessor.extract_shape(regrd_txx, os.path.join(cfg['auxiliary_data_dir'], cfg['era_regridding_region']))
    # else:
    #     reg_cb_txx = esmvalcore.preprocessor.extract_region(regrd_txx, cfg['era_regridding_region'][0], cfg['era_regridding_region'][1], 
    #                     cfg['era_regridding_region'][2], cfg['era_regridding_region'][3])

    anomal_era_cb = esmvalcore.preprocessor.anomalies(era_rxNday_big_cube, 'full',
                        reference={'start_year': cfg['reference_period'][0],
                        'start_month': 1, 'start_day':1,
                        'end_year': cfg['reference_period'][1],
                        'end_month': 12, 'end_day':31} )
    crop_era_cb = esmvalcore.preprocessor.extract_time(anomal_era_cb,
                        start_year=cfg['era_plotting_period'][0],
                        start_month=1, start_day=1,
                        end_year=cfg['era_plotting_period'][1],
                        end_month=12, end_day=31)

    return crop_era_cb 


def bootstrap_gev(data_dic): 

    # determining the max length of the model realisation, the number of bootstrap
    # iterations is this value * 100
    max_cblst_len = np.asarray([len(data_dic[model]) for model in data_dic.keys()]).max()
    iter_pool = max_cblst_len *100

    shapes = np.zeros(iter_pool)
    locs = np.zeros(iter_pool)
    scales = np.zeros(iter_pool)

    for i in range(0, iter_pool):
        pool_data = list()
        for model in data_dic.keys(): 
            n_real = len(data_dic[model])
            idx = np.random.default_rng().integers(low=0, high=n_real, size=1)[0]
            pool_data.append(data_dic[model][idx].data.data.round(2))
        pool_data = np.asarray(pool_data).flatten()
        shapes[i], locs[i], scales[i] = gev.fit(pool_data)
    

    param_dic = {'shape': shapes, 'loc': locs, 'scale': scales}    

    return param_dic


def make_uncert_figures(data_dic, cfg, border):

   # era_cb = data_dic['reanalysis']

    colors = {}
    colors['all'] = (196 / 255, 121 / 255, 0)
    colors['nat'] = (0, 79 / 255, 0)
    colors['ssp245'] = (69 / 255, 118 / 255, 191 / 255)

    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis') 

    x_gev = np.arange(-border,border+0.1, 1)

    tlocs = {'all': 0.04 , 'nat': 0.01,  'ssp245': 0.08}
  
    # this is a figure where we will plot single distributions from bootstrap
    fig_single_bootstrap, ax_single_bootstrap = plt.subplots(1)
    fig_single_bootstrap.set_size_inches(12., 8.)

    # this a figure where we plot how the values for GEV params are distributed 
    fig_gev_distr, ax_gev_distr = plt.subplots(3)
    fig_gev_distr.set_size_inches(8., 12.)

    uncert_band = {}
    for exp in exp_list: 
        gev_dic = data_dic[exp].pop('GEV_uncert')
        # here we plot distribution of single GEV params
        gev_params = ['shape', 'loc', 'scale']
        for n, gev_param in enumerate(gev_params):
            ax_gev_distr[n].hist(gev_dic[gev_param], bins=50, edgecolor='none',
                    facecolor = colors[exp], alpha=0.3, label = exp, density=True)
            ax_gev_distr[n].set_xlabel(gev_param)
            ax_gev_distr[n].set_ylabel('Number density')
            ax_gev_distr[n].set_title('GEV parameter: ' + gev_param)

        # this is an array where we'll throw all pdfs, to calculate 5/95 perc later 
        all_pdfs = np.zeros((len(x_gev), len(gev_dic['shape'])))

        # here we plot single distributions
        for i in range(len(gev_dic['shape'])):
            gev_pdf = gev.pdf(x_gev, gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
            all_pdfs[:, i] = gev_pdf
            ax_single_bootstrap.plot(x_gev, gev_pdf, color = colors[exp], alpha=0.03)
        
        uncert_band[exp] = {'x_gev': x_gev, '5th_perc' : np.percentile(all_pdfs, 5, axis = 1), '95th_perc': np.percentile(all_pdfs, 95, axis = 1)}
  
        param_str = '                             '+exp\
            + '\nshape mean:'+str(np.around(gev_dic['shape'].mean(),3))+', max:'+str(np.around(gev_dic['shape'].max(),3)) + ', min:' + str(np.around(gev_dic['shape'].min(),3)) \
            + '\n loc mean:'+str(np.around(gev_dic['loc'].mean(),3))+', max:'+str(np.around(gev_dic['loc'].max(),3)) + ', min:' + str(np.around(gev_dic['loc'].min(),3)) \
            + '\nscale mean:'+ str(np.around(gev_dic['scale'].mean(),3))+', max:'+str(np.around(gev_dic['scale'].max(),3)) + ', min:' + str(np.around(gev_dic['scale'].min(),3))               
        
        ax_single_bootstrap.text(-0.95*border, tlocs[exp], param_str, color = colors[exp])

    n_bins = np.around(np.arange(-border, border +0.1, 2))
    # era_hist = np.histogram(era_cb.data, bins=n_bins, density=True)
    # era_hist[0][era_hist[0]==0] = np.nan
    # ax_single_bootstrap.scatter(era_hist[1][:-1] + np.diff(era_hist[1])/2, era_hist[0], marker='_', c = 'r', label = 'ERA5', s=100, lw = 2.5) 

    ax_gev_distr[0].legend(loc=0, fancybox=False, frameon=False)
    fig_gev_distr.suptitle('Distribution of GEV parameters after bootstrap',
                fontsize = 'x-large')
    fig_gev_distr.set_dpi(250)
    plt.tight_layout()
    fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_exreme_distr_param' + diagtools.get_image_format(cfg)))
    fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_exreme_distr_param.png'))
    plt.close(fig_gev_distr)

    ax_single_bootstrap.legend(loc=2, fancybox=False, frameon=False)
    ax_single_bootstrap.set_xlim(-border, border)
    ax_single_bootstrap.set_ylim(0, ax_single_bootstrap.get_ylim()[1])
    ax_single_bootstrap.set_xlabel('Temperature anomaly, C')
    ax_single_bootstrap.set_ylabel('Number density')

    fig_single_bootstrap.suptitle('Estimation of GEV fit uncertainty from Multi Model Mean with Bootstrap method',
                fontsize = 'x-large')
    fig_single_bootstrap.set_dpi(250)

    fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_extremes_bootstrap'+ diagtools.get_image_format(cfg)))
    fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_extremes_bootstrap.png'))

    # ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_bootstrap')
    # ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_bootstrap',  img_ext='.png')
    plt.close(fig_single_bootstrap)
                            
    return uncert_band


def make_hist_figure(data_dic, cfg, uncert_band, border):

    era_cb = data_dic['reanalysis']
    era_max = era_cb.data.max()
    year_max = 2000+era_cb.data.argmax()
    era_2021 = era_cb.data[-1]
    
    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis')    

    colors = {'all' : (196 / 255, 121 / 255, 0), 
              'nat' : (0, 79 / 255, 0), 
              'ssp245' : (69 / 255, 118 / 255, 191 / 255)}

    tlocs = {'all': 0.02 , 'nat': 0.005,  'ssp245': 0.035}

    csv_file = open(os.path.join(cfg['work_dir'], 'gev_parameters.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['shape_'+exp_key+', loc_'+exp_key+ ', scale_'+exp_key for exp_key in exp_list]
    head_row.insert(0, 'model')
    gevs_csv_writer.writerow(head_row)
    models = data_dic[exp_list[0]].keys()
    for model in models: 
        fig = plt.figure()
        fig.set_size_inches(12., 8.)
        model_row = [model]
        for exp_key in exp_list:
            ens_cubelist = data_dic[exp_key][model]
            distrib_data = []
            weights = []
            for cube in ens_cubelist:
                if model == 'Multi-Model-Mean':
                    cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                else:
                    cube_weight = cube.attributes['ensemble_weight']
                for point in cube.data:
                    distrib_data.append(np.around(point, 2))
                    weights.append(cube_weight/len(cube.data))
            distrib_data = np.asarray(distrib_data)
            weights = np.asarray(weights)
            un_weights = np.unique(weights)
            rev_un_weights = np.asarray(1/un_weights).round(0).astype('int32')
            large_denom = np.gcd.reduce(rev_un_weights)
            dev_weights = rev_un_weights/large_denom
            least_mult = np.lcm.reduce(dev_weights.astype('int32'))
            un_factors = least_mult/dev_weights
            factors = np.zeros(len(weights))
            for n_w, un_wght in enumerate(un_weights):
                factors[np.where(weights==un_wght)] = un_factors[n_w]
            factors = factors.astype('int32')
            upd_distr_data = list()
            new_weights = list()
            for n_dp, distrib_point in enumerate(distrib_data): 
                for f in range(factors[n_dp]):
                    upd_distr_data.append(distrib_point)
                    new_weights.append(weights[n_dp]/factors[n_dp])
            upd_distr_data = np.asarray(upd_distr_data)
            if model == 'Multi-Model-Mean':
                print('Liza')
            w_shape, w_loc, w_scale = gev.fit(upd_distr_data)
            x_gev = uncert_band[exp_key]['x_gev']
            w_pdf = gev.pdf(x_gev, w_shape, w_loc, w_scale)
            model_row.extend([w_shape, w_loc, w_scale])
            n_bins = np.around(np.arange(-border, border + 0.1, 5))
            plt.hist(distrib_data, bins=n_bins, edgecolor=colors[exp_key],
                    facecolor = colors[exp_key], alpha=0.3, label=exp_key + ' N_real=' +str(len(ens_cubelist)), density=True, weights=weights) 
            plt.plot(x_gev, w_pdf, c = colors[exp_key], ls = 'solid', label = 'Fitted GEV '+exp_key)
            plt.text(135, tlocs[exp_key], '  ' +exp_key+' GEV\nshape='+str(np.around(w_shape,3))+ '\n loc='+ str(np.around(w_loc,3)) \
                   + '\nscale='+str(np.around(w_scale,3)), color= colors[exp_key])
            if model == 'Multi-Model-Mean':
                perc_5 = uncert_band[exp_key]['5th_perc']
                perc_95 = uncert_band[exp_key]['95th_perc']
                plt.fill_between(x_gev, perc_5, perc_95, color= colors[exp_key], alpha = 0.3, linewidth=0)
        gevs_csv_writer.writerow(model_row)
       # era_hist = np.histogram(era_cb.data, bins=n_bins, density=True)
       #  era_hist[0][era_hist[0]==0] = np.nan
       #  plt.scatter(era_hist[1][:-1] + np.diff(era_hist[1])/2, era_hist[0], marker='_', c = 'r', label = 'ERA5', s=100, lw = 2.5) 

        plt.scatter(era_2021, 0.04, s = 100, marker='o', c='r', label ='ERA5 (2021)')
        plt.scatter(era_max, 0.04, s = 100, marker='*', c='r', label ='ERA5 max ('+ str(year_max)+')')

        plt.legend(loc=2, fancybox=False, frameon=False)
        plt.xlim(-75, 175)
        plt.xlabel(cfg['ax_var_label']+' anomaly, ' +cfg['un_label'])
        plt.ylabel('Number density')

        fig.suptitle('Distribution of '+cfg['title_var_label']+' anomalies in '+cfg['region']+' \nrelative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+model, fontsize = 'x-large')
        fig.set_dpi(250)

        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+model)
        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+model, img_ext='.png')

    return

def make_timeseries_figure(data_dic, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    era_cb = data_dic['reanalysis']

    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis') 

    colors = {}
    colors['all'] = (196 / 255, 121 / 255, 0)
    colors['nat'] = (0, 79 / 255, 0)
    colors['ssp245'] = (69 / 255, 118 / 255, 191 / 255)

    models = data_dic[exp_list[0]].keys()
    for model in models:
        fig = plt.figure()
        fig.set_size_inches(12., 8.)
        plt.hlines(0, -200, 200, colors = 'grey', linestyles='dashed')
        for exp_key in exp_list:
            ens_cubelist = data_dic[exp_key][model]
            time_data = []
            weights = []
            for cube in ens_cubelist:
                if model == 'Multi-Model-Mean':
                    cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
                else:
                    cube_weight = cube.attributes['ensemble_weight']
                weights.append([cube_weight]*len(cube.data))
                time_data.append(cube.data)
            time_data = np.asarray(time_data)
            weights = np.asarray(weights)
            tim_coord = iris.coords.DimCoord(np.arange(0, len(cube.coord('time').points)), long_name = 'n_years', var_name = 'n_years')
            aux_coord = iris.coords.DimCoord(np.arange(0, len(ens_cubelist)), standard_name=None, long_name='aux_ens_coord', var_name='aux_ens_coord')
            full_cube = iris.cube.Cube(time_data, long_name =cube.long_name, var_name = cube.var_name, units = cube.units,  dim_coords_and_dims = [(aux_coord, 0),(tim_coord, 1)])
            if len(ens_cubelist) > 1:
                mean_cb = full_cube.collapsed('aux_ens_coord', iris.analysis.MEAN, weights = weights)
                perc_cb = full_cube.collapsed('aux_ens_coord', iris.analysis.WPERCENTILE, percent=[5,95], weights = weights)
                plt.fill_between(tim_coord.points, perc_cb.data[0,:], perc_cb.data[1,:], color=colors[exp_key], alpha=0.1, lw=0)
            else: 
                mean_cb = cube
            plt.plot(tim_coord.points, mean_cb.data, c=colors[exp_key], label = exp_key, lw=1.5)
        plt.plot(np.arange(0, len(era_cb.coord('time').points)), era_cb.data, lw=1.5, c='r', label = 'ERA5')

        plt.legend(loc=2, fancybox=False, frameon=False)
        plt.xlim(-1,len(era_cb.coord('time').points))
        plt.ylabel(cfg['ax_var_label']+' anomaly, ' + cfg['un_label'])
        plt.xlabel('Years since the start of a dataset')

        fig.suptitle('Timeseries of '+cfg['title_var_label']+' anomalies in '+cfg['region']+' \nrelative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+' from ' + model, fontsize = 'x-large')
        fig.set_dpi(250)
        
        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_timeseries_'+model)
        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_timeseries_'+model, img_ext='.png')   

    return


def main(cfg):

    anomaly_rxNday(cfg)

    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)

    groups_l = list(groups.keys()) ; groups_l.remove('anomaly')

    era_cube = get_era_txx(cfg)

    mins = list(); maxs = list()

    plotting_dic = {}

    for group in groups_l:
        plotting_dic[group] = {}
        group_data = groups[group]
        datasets = group_metadata(group_data, 'dataset')
        ens_cubelist = iris.cube.CubeList()
        for dataset in datasets.keys():
            filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
            n_real = len(filepaths)
            mod_cubelist = iris.cube.CubeList()
            for filepath in filepaths:
                mod_cb = iris.load_cube(filepath)
                file_metadata = select_metadata(datasets[dataset], filename = filepath)
                ens = file_metadata[0]['ensemble']
                if ( cfg['mult_factor'] != 1)&(group != 'obs'):
                    mod_cb = mod_cb * cfg['mult_factor']
                rxNday_cb = mod_cb.rolling_window('time', iris.analysis.SUM, cfg['n_days'])
                rxNday_cb = esmvalcore.preprocessor.annual_statistics(rxNday_cb, operator='max')
                anom_cb = iris.load_cube(select_metadata(input_data.values(), dataset = dataset, ensemble = ens, variable_group = 'anomaly')[0]['filename'])
                rxNday_ano_cb = (rxNday_cb - anom_cb)
                mins.append(rxNday_ano_cb.collapsed('time', iris.analysis.MIN).data)
                maxs.append(rxNday_ano_cb.collapsed('time', iris.analysis.MAX).data)
                rxNday_ano_cb.attributes['ensemble_weight'] = 1 / n_real
                rxNday_ano_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
                # provenance_rec= { 'authors' : 'malinina_elizaveta', 'statistics': 'max', 'ancestors': [filepath]}
                ens_cubelist.append(rxNday_ano_cb)
                mod_cubelist.append(rxNday_ano_cb)
            plotting_dic[group][dataset] = mod_cubelist
        if group != 'obs':     
            plotting_dic[group]['GEV_uncert'] = bootstrap_gev(plotting_dic[group]) 
            plotting_dic[group]['Multi-Model-Mean'] = ens_cubelist
    
    plotting_dic['reanalysis'] = era_cube

    min_var = np.asarray(mins).min() ; max_var = np.asarray(maxs).max()  

    border = np.ceil(np.max(np.abs([min_var, max_var]))*1.75)

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    uncert_band = make_uncert_figures(plotting_dic, cfg, border)

    make_hist_figure(plotting_dic, cfg, uncert_band, border)

    make_timeseries_figure(plotting_dic, cfg)                                   

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
