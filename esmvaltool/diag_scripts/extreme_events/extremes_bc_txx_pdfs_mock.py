import csv
from math import dist
import esmvalcore.preprocessor
import iris
from iris.util import equalise_attributes
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r as gumbel

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
        if cfg.get('calculate_api'): 
            k = cfg['api_k']
            n_arr = len(ano_cb.coord('time').points) - cfg['n_days']
            api_data = np.zeros(n_arr)
            for i in range(0,n_arr):
                api_data[i] = np.sum([(k**t)*ano_cb.data[i+cfg['n_days'] - t] for t in range(1,cfg['n_days']+1)])
            t_coord = iris.coords.DimCoord(ano_cb.coord('time').points[cfg['n_days']:], bounds=ano_cb.coord('time').bounds[cfg['n_days']:,:], 
                      long_name=ano_cb.coord('time').long_name, standard_name=ano_cb.coord('time').standard_name, units=ano_cb.coord('time').units,
                      var_name=ano_cb.coord('time').var_name)
            rxNd_cube = iris.cube.Cube(api_data, dim_coords_and_dims=[(t_coord,0)], long_name='Antecedent Precipitation Index', var_name='api', 
                        units=ano_cb.units,attributes=ano_cb.attributes)
        else:
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
        if cfg['era_regridding_shape']:
            reg_cube = esmvalcore.preprocessor.extract_shape(regrid_cube, os.path.join(aux_dir,cfg['era_shape_file']), method='contains', crop=True)
        else:
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

    if cfg.get('calculate_api'): 
        k = cfg['api_k']
        n_arr = len(era_big_cube.coord('time').points) - cfg['n_days']
        api_data = np.zeros(n_arr)
        for i in range(0,n_arr):
            api_data[i] = np.sum([(k**t)*era_big_cube.data[i+cfg['n_days'] - t] for t in range(1,cfg['n_days']+1)])
        t_coord = iris.coords.DimCoord(era_big_cube.coord('time').points[cfg['n_days']:], bounds=era_big_cube.coord('time').bounds[cfg['n_days']:,:], 
                    long_name=era_big_cube.coord('time').long_name, standard_name=era_big_cube.coord('time').standard_name, 
                    units=era_big_cube.coord('time').units, var_name=era_big_cube.coord('time').var_name)
        era_rxNday_big_cube = iris.cube.Cube(api_data, dim_coords_and_dims=[(t_coord,0)], long_name='Antecedent Precipitation Index', var_name='api', 
                    units=era_big_cube.units,attributes=era_big_cube.attributes)
    else:
        era_rxNday_big_cube = era_big_cube.rolling_window('time', iris.analysis.SUM, cfg['n_days'])

    era_rxNday_big_cube = esmvalcore.preprocessor.annual_statistics(era_rxNday_big_cube, 'max')  

    ref_era_cb = esmvalcore.preprocessor.extract_time(era_rxNday_big_cube,
                        start_year=cfg['reference_period'][0],
                        start_month=1, start_day=1,
                        end_year=cfg['reference_period'][1]+1,
                        end_month=1, end_day=1)

    ref_era_cb = esmvalcore.preprocessor.climate_statistics(ref_era_cb, operator='mean',
                                                           period='full')


    use_era_cb = era_rxNday_big_cube / ref_era_cb 

    return use_era_cb 


def bootstrap_gev(data_dic, distrib = 'gev', yblock = 5): 

    # determining the max length of the model realisation, the number of bootstrap
    # iterations is this value * 100
    max_cblst_len = np.asarray([len(data_dic[model]) for model in data_dic.keys()]).max()
    iter_pool = max_cblst_len *100
    n_real = len(data_dic.keys())
    n_years = np.asarray([data_dic[model][0].shape for model in data_dic.keys()]).max()

    pool_data = list()
    for model in data_dic.keys():
        for i in range(len(data_dic[model])):
            pool_data.append(data_dic[model][i].data)
    pool_data = np.asarray(pool_data)

    shapes = np.zeros(iter_pool)
    locs = np.zeros(iter_pool)
    scales = np.zeros(iter_pool)

    for i in range(0, iter_pool):
        sample_data = list()
        mod_idxs = np.random.randint(0, high = pool_data.shape[0], size=np.int(np.around(n_real*n_years/yblock)))
        for mod_idx in mod_idxs: 
            y_idx = np.random.randint(0, high=n_years-yblock)
            sample_data.append(pool_data[mod_idx, y_idx:y_idx+yblock])
        sample_data = np.asarray(sample_data).flatten()
        if distrib.lower()=='gev':
            shapes[i], locs[i], scales[i] = gev.fit(sample_data, method='MLE')
        elif distrib.lower()=='gumbel':
            locs[i], scales[i] = gumbel.fit(sample_data, method='MLE')
    
    param_dic = {'loc': locs, 'scale': scales}    
    if distrib.lower()=='gev': 
        param_dic['shape'] = shapes

    return param_dic


def make_uncert_figures(data_dic, cfg, border, distrib='gev'):

    colors = {}
    colors['all'] = (196 / 255, 121 / 255, 0)
    colors['nat'] = (0, 79 / 255, 0)
    colors['ssp245'] = (69 / 255, 118 / 255, 191 / 255)

    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis') 

    x_gev = np.arange(border[0],border[1], 0.01)

    tlocs = {'all': 0.04 , 'nat': 0.01,  'ssp245': 0.08}
  
    # this is a figure where we will plot single distributions from bootstrap
    fig_single_bootstrap, ax_single_bootstrap = plt.subplots(1)
    fig_single_bootstrap.set_size_inches(12., 8.)

    if distrib.lower() == 'gev': 
        gev_params = ['shape', 'loc', 'scale']
    elif distrib.lower() == 'gumbel':
        gev_params = ['loc', 'scale']

    # this a figure where we plot how the values for GEV params are distributed 
    fig_gev_distr, ax_gev_distr = plt.subplots(len(gev_params))
    fig_gev_distr.set_size_inches(8., 12.)

    uncert_band = {}
    for exp in exp_list: 
        gev_dic = data_dic[exp].pop('GEV_uncert')
        # here we plot distribution of single GEV params
        for n, gev_param in enumerate(gev_params):
            ax_gev_distr[n].hist(gev_dic[gev_param], bins=50, edgecolor='none',
                    facecolor = colors[exp], alpha=0.3, label = exp, density=True)
            ax_gev_distr[n].set_xlabel(gev_param)
            ax_gev_distr[n].set_ylabel('Number density')
            ax_gev_distr[n].set_title(distrib +' parameter: ' + gev_param)

        # this is an array where we'll throw all pdfs, to calculate 5/95 perc later 
        all_pdfs = np.zeros((len(x_gev), len(gev_dic['loc'])))
        all_sfs = np.zeros((len(x_gev), len(gev_dic['loc'])))

        # here we plot single distributions
        for i in range(len(gev_dic['loc'])):
            if distrib.lower() == 'gev':
                gev_pdf = gev.pdf(x_gev, gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
                gev_sf = gev.sf(x_gev, gev_dic['shape'][i],gev_dic['loc'][i], gev_dic['scale'][i])
            elif distrib.lower() == 'gumbel': 
                gev_pdf = gumbel.pdf(x_gev, gev_dic['loc'][i], gev_dic['scale'][i])
                gev_sf = gumbel.sf(x_gev, gev_dic['loc'][i], gev_dic['scale'][i])
            all_pdfs[:, i] = gev_pdf
            all_sfs[:, i] = gev_sf
            ax_single_bootstrap.plot(x_gev, gev_pdf, color = colors[exp], alpha=0.03)
        
        uncert_band[exp] = {'x_gev': x_gev, 'pdf_5th_perc' : np.percentile(all_pdfs, 5, axis = 1), 
                                            'pdf_95th_perc': np.percentile(all_pdfs, 95, axis = 1),
                                            'sf_5th_perc' : np.percentile(all_sfs, 5, axis = 1), 
                                            'sf_95th_perc' : np.percentile(all_sfs, 95, axis = 1)}
  
        param_str = '                             '+exp
        if distrib.lower() == 'gev':
            param_str += '\nshape mean:'+str(np.around(gev_dic['shape'].mean(),3))+', max:'+str(np.around(gev_dic['shape'].max(),3)) + ', min:' + str(np.around(gev_dic['shape'].min(),3)) 
        param_str += '\n loc mean:'+str(np.around(gev_dic['loc'].mean(),3))+', max:'+str(np.around(gev_dic['loc'].max(),3)) + ', min:' + str(np.around(gev_dic['loc'].min(),3)) \
            + '\nscale mean:'+ str(np.around(gev_dic['scale'].mean(),3))+', max:'+str(np.around(gev_dic['scale'].max(),3)) + ', min:' + str(np.around(gev_dic['scale'].min(),3))               
        
        ax_single_bootstrap.text(-0.95*border[0], tlocs[exp], param_str, color = colors[exp])

    ax_gev_distr[0].legend(loc=0, fancybox=False, frameon=False)
    fig_gev_distr.suptitle('Distribution of ' + distrib+' parameters after bootstrap',
                fontsize = 'x-large')
    fig_gev_distr.set_dpi(250)
    plt.tight_layout()
    fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_exreme_distr_param_'+distrib.lower() + diagtools.get_image_format(cfg)))
    fig_gev_distr.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_exreme_distr_param_'+distrib.lower()+'.png'))
    plt.close(fig_gev_distr)

    ax_single_bootstrap.legend(loc=2, fancybox=False, frameon=False)
    ax_single_bootstrap.set_xlim(border[0], border[1])
    ax_single_bootstrap.set_ylim(0, ax_single_bootstrap.get_ylim()[1])
    ax_single_bootstrap.set_xlabel('Precipitation ratio, mm')
    ax_single_bootstrap.set_ylabel('Number density')

    fig_single_bootstrap.suptitle('Estimation of ' + distrib +' fit uncertainty from Multi Model Mean with Bootstrap method',
                fontsize = 'x-large')
    fig_single_bootstrap.set_dpi(250)

    fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_extremes_bootstrap_'+distrib.lower() + diagtools.get_image_format(cfg)))
    fig_single_bootstrap.savefig(os.path.join(cfg['plot_dir'], 'figure_bc_extremes_bootstrap_'+distrib.lower() +'.png'))
    plt.close(fig_single_bootstrap)
                            
    return uncert_band


def make_hist_figure(data_dic, cfg, uncert_band, border, apr_param, distrib = 'gev'):

    era_cb = data_dic['reanalysis']
    era_max = era_cb.data.max()
    year_max = 1950+era_cb.data.argmax()
    era_2021 = era_cb.data[-1]
    if distrib.lower() == 'gev':
        era_gev_params = gev.fit(era_cb.data)
    elif distrib.lower() == 'gumbel':
        era_gev_params = gumbel.fit(era_cb.data)

    era_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_era_data.csv'), 'w', newline='')
    era_csv_writer = csv.writer(era_csv, delimiter=',')
    era_csv_writer.writerow(['2021 ERA value '+str(era_2021)])
    era_csv_writer.writerow(['Max ERA value '+str(era_max)+ ' in '+ str(year_max)])
    era_csv_writer.writerow(['ERA ' +distrib +' params'])
    if distrib.lower() == 'gev':
        era_csv_writer.writerow(['shape', 'loc', 'scale'])
    elif distrib.lower() == 'gumbel':
        era_csv_writer.writerow(['loc', 'scale'])
    era_csv_writer.writerow(era_gev_params)
    era_csv.close()

    risk_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_risk_data.csv'), 'w', newline='')
    risk_csv_writer = csv.writer(risk_csv, delimiter=',')
    risk_head_row = ['model']

    exp_list = list(data_dic.keys()) ; exp_list.remove('reanalysis')    

    quantile_measures = np.arange(0, 1.01, 0.01); quantile_measures[0] = 0.001

    colors = {'all' : (196 / 255, 121 / 255, 0), 
              'nat' : (0, 79 / 255, 0), 
              'ssp245' : (69 / 255, 118 / 255, 191 / 255)}

    tlocs = {'all': 0.55 , 'nat': 0.15,  'ssp245': 0.95}

    csv_file = open(os.path.join(cfg['work_dir'], distrib.lower()+'_parameters.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['model']
    for exp_key in exp_list:
        if distrib.lower() == 'gev':
            head_row.append('shape_'+exp_key) 
        head_row.append('loc_'+exp_key); head_row.append('scale_'+exp_key)
        risk_head_row.append(exp_key+'_prob'); risk_head_row.append(exp_key+'_return_p')
    gevs_csv_writer.writerow(head_row)
    risk_csv_writer.writerow(risk_head_row)
    models = data_dic[exp_list[0]].keys()
    for model in models: 
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(12., 8.)
        gs = fig.add_gridspec(nrows=2, ncols=6)
        ax_hist = fig.add_subplot(gs[:, :-2])
        ax_qq= fig.add_subplot(gs[0, -2:])
        ax_surv= fig.add_subplot(gs[1, -2:])
        model_row = [model]
        risk_model_row = [model]
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
            x_gev = uncert_band[exp_key]['x_gev']
            if distrib.lower() == 'gev':
                w_distr_par = gev.fit(upd_distr_data, loc=apr_param[exp_key]['loc'], scale=apr_param[exp_key]['scale'], method='MLE')
                w_pdf = gev.pdf(x_gev, *w_distr_par)
                w_survival = gev.sf(x_gev, *w_distr_par)
                theor_quants = gev(*w_distr_par).ppf(quantile_measures)
            elif distrib.lower() == 'gumbel':
                w_distr_par = gumbel.fit(upd_distr_data, loc=apr_param[exp_key]['loc'], method='MLE')
                w_pdf = gumbel.pdf(x_gev, *w_distr_par)
                w_survival = gumbel.sf(x_gev, *w_distr_par)
                theor_quants = gumbel(*w_distr_par).ppf(quantile_measures)
            model_row.extend(w_distr_par)
            n_bins = np.arange(int(border[0]*20)/20, border[1]+0.1, 0.1)
            ax_hist.hist(distrib_data, bins=n_bins, edgecolor=colors[exp_key],
                    facecolor = colors[exp_key], alpha=0.3, label=cfg['name_' + exp_key] , density=True, weights=weights, zorder = 2)
            ax_hist.plot(x_gev, w_pdf, c = colors[exp_key], ls = 'solid', label = distrib+' fit '+cfg['name_' + exp_key], zorder=3)
            if model == 'Multi-Model-Mean':
                pdf_perc_5 = uncert_band[exp_key]['pdf_5th_perc']
                pdf_perc_95 = uncert_band[exp_key]['pdf_95th_perc']
                sf_perc_5 = uncert_band[exp_key]['sf_5th_perc']
                sf_perc_95 = uncert_band[exp_key]['sf_95th_perc']
                ax_hist.fill_between(x_gev, pdf_perc_5, pdf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
                ax_surv.fill_between(x_gev, 1/sf_perc_5, 1/sf_perc_95, color=colors[exp_key], alpha=0.3, linewidth=0, zorder=4)
            event_idx = np.argmin(np.abs(x_gev - era_2021))
            max_idx = np.argmin(np.abs(x_gev - era_max))
            risk_model_row.extend([w_survival[event_idx], 1/w_survival[event_idx]])
            pract_quants = np.quantile(upd_distr_data, quantile_measures)
            ax_qq.scatter(theor_quants, pract_quants, edgecolors=colors[exp_key], marker='o', facecolors='None', lw=0.75, label=cfg['name_' + exp_key], zorder=3)
            ax_surv.plot(x_gev, 1/w_survival, color=colors[exp_key], zorder=3)
        gevs_csv_writer.writerow(model_row)
        risk_csv_writer.writerow(risk_model_row)

        ylims = ax_hist.get_ylim()
        ax_hist.set_ylim(*ylims)

        ax_hist.text(2.4, ylims[1]*0.65,'  Number of\nrealisations ' +str(len(ens_cubelist)), fontsize='large')
        ax_hist.vlines(era_2021, *ylims, color = 'indianred', linestyle = 'solid', lw=1.5, zorder=1, label = 'ERA5 (2021)')
        ax_hist.vlines(era_max, *ylims, color = 'indianred', linestyle = 'dashed', lw=1.5, zorder=1, label = 'ERA5 max ('+ str(year_max)+')')

        if distrib.lower() == 'gev':
            era_surv = gev.sf(x_gev, *era_gev_params)
        elif distrib.lower() == 'gumbel': 
            era_surv = gumbel.sf(x_gev, *era_gev_params)
        era_event_prob = era_surv[event_idx]
        era_max_prob = era_surv[max_idx]

        ax_surv.vlines(era_2021, 0.1, 1/era_event_prob, linestyle = 'solid', color='indianred', zorder=2,  label = 'ERA5 (2021)')
        ax_surv.vlines(era_max, 0.1, 1/era_max_prob, linestyle = 'dashed', color='indianred', zorder=1, label = 'ERA5 ('+ str(year_max)+')')
        ax_surv.hlines(1/era_event_prob, x_gev[0], era_2021,linestyle = 'solid', color='indianred', zorder=2)
        ax_surv.hlines(1/era_max_prob, x_gev[0], era_max,  linestyle = 'dashed', color='indianred', zorder=1)

        ax_hist.legend(loc=1, fancybox=False, frameon=False)
        ax_qq.legend(loc=2, fancybox=False, frameon=False, handletextpad=0.01)
        ax_qq.plot([0,5],[0,5], c='tab:grey', zorder=1)
        # plt.xlim(-0.5*border, 0.7*border)
        ax_hist.set_xlim(x_gev[0], 3)
        ax_qq.set_xlim(x_gev[0], 3)
        ax_qq.set_ylim(x_gev[0], 3)
        ax_surv.set_ylim(1,1000)
        ax_surv.set_xlim(x_gev[0], 3)
        ax_surv.set_yscale('log')
        ax_hist.set_title('Probability density function')
        ax_qq.set_title('Quality assessment')
        ax_qq.set_ylabel('Data quantile')
        ax_qq.set_xlabel(distrib+' quantile')
        ax_surv.set_title('Return period')
        ax_surv.set_ylabel('years')
        ax_surv.set_xlabel('Normalized '+cfg['ax_var_label'])
        ax_hist.set_xlabel('Normalized '+cfg['ax_var_label'])
        ax_hist.set_ylabel('Number density')

        if model == 'Multi-Model-Mean':
            fig.suptitle('Normalised '+cfg['title_var_label']+' in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+str(len(models) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig.suptitle('Normalised '+cfg['title_var_label']+' in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+model, fontsize = 'x-large')
        fig.set_dpi(250)

        plt.tight_layout()

        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+distrib.lower()+'_'+model)
        ipcc_sea_ice_diag.figure_handling(cfg, name='figure_bc_extremes_'+distrib.lower()+'_'+model, img_ext='.png')

    return


def main(cfg):

    anomaly_rxNday(cfg)

    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)

    groups_l = list(groups.keys()) ; groups_l.remove('anomaly')

    era_cube = get_era_txx(cfg)

    mins = list(); maxs = list()

    distrib_fit= cfg['fit_distribution']

    plotting_dic = {}

    fit_param_apr = {}

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
                if cfg.get('calculate_api'): 
                    k = cfg['api_k']
                    n_arr = len(mod_cb.coord('time').points) - cfg['n_days']
                    api_data = np.zeros(n_arr)
                    for i in range(0,n_arr):
                        api_data[i] = np.sum([(k**t)*mod_cb.data[i+cfg['n_days'] - t] for t in range(1,cfg['n_days']+1)])
                    t_coord = iris.coords.DimCoord(mod_cb.coord('time').points[cfg['n_days']:], bounds=mod_cb.coord('time').bounds[cfg['n_days']:,:], 
                            long_name=mod_cb.coord('time').long_name, standard_name=mod_cb.coord('time').standard_name, units=mod_cb.coord('time').units,
                            var_name=mod_cb.coord('time').var_name)
                    rxNday_cb = iris.cube.Cube(api_data, dim_coords_and_dims=[(t_coord,0)], long_name='Antecedent Precipitation Index', var_name='api', 
                                units=mod_cb.units,attributes=mod_cb.attributes)
                else:
                    rxNday_cb = mod_cb.rolling_window('time', iris.analysis.SUM, cfg['n_days'])
                rxNday_cb = esmvalcore.preprocessor.annual_statistics(rxNday_cb, operator='max')
                anom_cb = iris.load_cube(select_metadata(input_data.values(), dataset = dataset, ensemble = ens, variable_group = 'anomaly')[0]['filename'])
                rxNday_ano_cb = rxNday_cb / anom_cb
                mins.append(rxNday_ano_cb.collapsed('time', iris.analysis.MIN).data)
                maxs.append(rxNday_ano_cb.collapsed('time', iris.analysis.MAX).data)
                rxNday_ano_cb.attributes['ensemble_weight'] = 1 / n_real
                rxNday_ano_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
                ens_cubelist.append(rxNday_ano_cb)
                mod_cubelist.append(rxNday_ano_cb)
            plotting_dic[group][dataset] = mod_cubelist
        plotting_dic[group]['GEV_uncert'] = bootstrap_gev(plotting_dic[group], distrib=distrib_fit, yblock = cfg['yblock']) 
        plotting_dic[group]['Multi-Model-Mean'] = ens_cubelist
        fit_param_apr[group] = {'loc': np.around(plotting_dic[group]['GEV_uncert']['loc'].mean(),3),
                                'scale': np.around(plotting_dic[group]['GEV_uncert']['scale'].mean(),3)}
        if distrib_fit.lower() == 'gev':
            fit_param_apr[group]['shape'] = np.around(plotting_dic[group]['GEV_uncert']['shape'].mean(),3)
    
    plotting_dic['reanalysis'] = era_cube

    min_var = np.asarray(mins).min() ; max_var = np.asarray(maxs).max()  

    border = [np.around(min_var*0.25,2), np.around(max_var*1.75,2)]

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    uncert_band = make_uncert_figures(plotting_dic, cfg, border, distrib = distrib_fit)

    make_hist_figure(plotting_dic, cfg, uncert_band, border, fit_param_apr, distrib = distrib_fit)

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
