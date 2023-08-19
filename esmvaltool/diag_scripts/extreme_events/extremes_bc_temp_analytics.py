import csv
import esmvalcore.preprocessor as eprep
import iris
from iris.util import equalise_attributes
import iris.plot as iplt
import cf_units
import cftime 
import cartopy.crs as ccrs 
import climextremes as cex
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.extreme_events.extremes_bc_temperature import bootstrap_gev
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def bootstrap_gev(var_data_dic, obs_gev_data, gsat_data_cblst=None): 

    if type(var_data_dic) == dict:  
        # determining the max length of the model realisation, the number of bootstrap
        # iterations is this value * 100
        max_cblst_len = np.asarray([len(var_data_dic[model]['var_data']) for model in var_data_dic.keys()]).max()
        iter_pool = max_cblst_len *100
        n_real = len(var_data_dic.keys())
        n_years = np.asarray([var_data_dic[model]['var_data'][0].shape for model in var_data_dic.keys()]).max()
        pool_size = int(np.around(n_real*n_years))
    elif type(var_data_dic) == iris.cube.CubeList: 
        iter_pool = np.max([len(var_data_dic) *100, 1000])
        n_years = var_data_dic[0].shape[0]
        n_real = 3 # this is a number of realisations which shown to be enough to draw conclusions
        if len(var_data_dic)<n_real: 
            pool_size = len(var_data_dic) * n_years
        else: 
            pool_size = n_real * n_years

    pool_data = list()
    pool_gsat_data = list()
    if type(var_data_dic) == dict: 
        for model in var_data_dic.keys():
            for i in range(len(var_data_dic[model]['var_data'])):
                pool_data.append(var_data_dic[model]['var_data'][i].data)
                pool_gsat_data.append(var_data_dic[model]['gsat_data'][i].data)
    elif type(var_data_dic) == iris.cube.CubeList:
        for cb in var_data_dic: 
            pool_data.append(cb.data)
        for gcb in gsat_data_cblst:
            pool_gsat_data.append(gcb.data)
    pool_data = np.asarray(pool_data)
    pool_gsat_data = np.asarray(pool_gsat_data) 

    shapes = np.zeros(iter_pool)
    locs = np.zeros(iter_pool)
    scales = np.zeros(iter_pool)
    rps  = np.zeros(iter_pool)

    rng = np.random.default_rng(501)

    for i in range(0, iter_pool):
        sample_data = list()
        sample_gsat_data = list()
        mod_idxs = rng.choice(pool_data.shape[0], size = pool_size)
        for mod_idx in mod_idxs: 
            y_idx = rng.choice(n_years)
            sample_data.append(pool_data[mod_idx, y_idx])
            sample_gsat_data.append(pool_gsat_data[mod_idx, y_idx])
        sample_data = np.asarray(sample_data).flatten()
        sample_gsat_data = np.asarray(sample_gsat_data).flatten()
        non_stat_params = cex.fit_gev(sample_data, sample_gsat_data, locationFun=1, getParams=True)
        try:
            shapes[i] = non_stat_params['mle'][3]; scales[i] = non_stat_params['mle'][2]
            locs[i]= non_stat_params['mle'][1]*obs_gev_data['ana_gsat_value'] + non_stat_params['mle'][0]
            rps[i] = np.around(1/gev.sf(obs_gev_data['ana_year_value'], -1*shapes[i],
                                        loc=locs[i], scale=scales[i]), 1)
        except: 
            shapes[i] = np.nan ; locs[i] = np.nan ; scales[i] = np.nan ; rps[i] = np.nan 
    
    param_dic = {'loc': locs, 'scale': scales, 'shape': shapes, 'return_periods': rps}

    return param_dic


def calculate_uncert_band(data_dic, mixns, cfg):

    border = [np.floor(mixns['min']*1.5), np.ceil(mixns['max']*1.5)]
    x_gev = np.arange(border[0],border[1], 0.1)

    uncert_band = {'x_gev': x_gev}

    for model in data_dic.keys():
        uncert_band[model] = {}
        mod_gevs = data_dic[model]['uncert']
        all_pdfs = np.zeros((len(x_gev), len(mod_gevs['loc'])))
        all_sfs = np.zeros((len(x_gev), len(mod_gevs['loc']))) 
        for i in range(len(mod_gevs['loc'])):
            gev_pdf = gev.pdf(x_gev, -1*mod_gevs['shape'][i],mod_gevs['loc'][i], mod_gevs['scale'][i])
            gev_sf = gev.sf(x_gev, -1*mod_gevs['shape'][i],mod_gevs['loc'][i], mod_gevs['scale'][i])
            all_pdfs[:, i] = gev_pdf
            all_sfs[:, i] = gev_sf
            uncert_band[model] = {'pdf_5th_perc' : np.nanpercentile(all_pdfs, 5, axis = 1), 
                                'pdf_95th_perc': np.nanpercentile(all_pdfs, 95, axis = 1),
                                'sf_5th_perc' : np.nanpercentile(all_sfs, 5, axis = 1), 
                                'sf_95th_perc' : np.nanpercentile(all_sfs, 95, axis = 1),
                                'sf_data_full' : all_sfs,
                                'return_periods_all': data_dic[model]['uncert']['return_periods'], 
                                'loc_min': mod_gevs['loc'].min(), 'loc_max': mod_gevs['loc'].max(),
                                'loc_5': np.nanpercentile(mod_gevs['loc'], 5, interpolation='nearest'),
                                'loc_95': np.nanpercentile(mod_gevs['loc'], 95, interpolation='nearest'),
                                'scale_min': mod_gevs['scale'].min(), 'scale_max': mod_gevs['scale'].max(),
                                'scale_5': np.nanpercentile(mod_gevs['scale'], 5, interpolation='nearest'),
                                'scale_95': np.nanpercentile(mod_gevs['scale'], 95, interpolation='nearest'),
                                'shape_min': mod_gevs['shape'].min(), 'shape_max': mod_gevs['shape'].max(),
                                'shape_5': np.nanpercentile(mod_gevs['shape'], 5, interpolation='nearest'),
                                'shape_95': np.nanpercentile(mod_gevs['shape'], 95, interpolation='nearest')}

    return uncert_band

def obtain_obs_info(groups, cfg):

    obs_txnx_info = groups.pop('obs_txnx')
    obs_gsat_info = groups.pop('obs_gsat')

    ano_obs_cb = iris.load_cube(obs_txnx_info[0]['filename'])
    ano_obs_arr = ano_obs_cb.data

    gsat_obs_cb = iris.load_cube(obs_gsat_info[0]['filename'])
    gsat_df = pd.DataFrame(gsat_obs_cb.data, columns=['gsat'])

    # smoothing gsat
    kernel_size = cfg['smooth_gsat_years']
    gsat_smooth_df = gsat_df.rolling(kernel_size, min_periods=1).mean()
    gsat_smooth_arr = gsat_smooth_df['gsat'].to_numpy()
    gsat_obs_cb.data = gsat_smooth_arr
    
    ana_year_const = iris.Constraint(time = lambda cell: cell.point.year == cfg['analysis_year'])
    ana_year_value =  float(ano_obs_cb.extract(ana_year_const).data)
    # determine the index of the ana_year_value, if several, assume the latest
    ana_arg = max(np.where(ano_obs_arr == ana_year_value)[0])
    ana_gsat = gsat_smooth_arr[ana_arg]

    # calculate stationary gev 
    obs_stat = cex.fit_gev(ano_obs_arr, returnValue=ana_year_value, getParams=True)
    orig_stat_rp = np.exp(obs_stat['logReturnPeriod'][0]) # it is the only value, we are making it a float

    # calculate non-stationary gev
    obs_non_stat = cex.fit_gev(ano_obs_arr, gsat_smooth_arr, locationFun=1, returnValue=ana_year_value, getParams=True)
    orig_nonstat_rp = np.exp(obs_non_stat['logReturnPeriod'][ana_arg])

    bootstrap_rps = list()
    rng = np.random.default_rng(501)

    for i in range(1000): 
        for_fit_indices = rng.integers(low=0, high=len(ano_obs_arr), size=len(ano_obs_arr))
        for_fit_indices.sort()
        temp_gev = cex.fit_gev(ano_obs_arr[for_fit_indices], gsat_smooth_arr[for_fit_indices],
                               locationFun=1, initial={'location':float(np.around(obs_stat['mle'][0],2)),
                                            'scale':float(np.around(obs_stat['mle'][1],2)), 
                                            'shape':float(np.around(obs_stat['mle'][2],2))}, getParams=True)
        try:
            temp_loc = temp_gev['mle'][0] + temp_gev['mle'][1]*ana_gsat
            temp_rp = np.around(1/gev.sf(ana_year_value, -1*temp_gev['mle'][3],
                                        loc= temp_loc, scale=temp_gev['mle'][2]), 1)
            bootstrap_rps.append(temp_rp)
        except:
            bootstrap_rps.append(np.nan)
    
    bootstrap_rps = np.asarray(bootstrap_rps)

    rp_perc = np.nanpercentile(bootstrap_rps, [5,10,50,90,95]).round(1)

    era_csv = open(os.path.join(cfg['work_dir'], 'analytics_gev_era_data_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    era_csv_writer = csv.writer(era_csv, delimiter=',')
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA5 '+cfg['ax_var_label']+' value '+str(ana_year_value)+ ', ERA5 smoothed GSAT value '+str(ana_gsat)])
    era_csv_writer.writerow(['ERA5 non-stationary GEV params'])
    era_csv_writer.writerow(list(obs_non_stat['mle_names']))
    era_csv_writer.writerow(list(obs_non_stat['mle']))
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA5 return period', str(np.around(orig_nonstat_rp,1))])
    era_csv_writer.writerow(['Bootstrapped uncertanties on ERA5 nonstationary return period'])
    era_csv_writer.writerow(['5_perc', '10_perc', '50_perc', '90_perc', '95_perc'])
    era_csv_writer.writerow(rp_perc)
    era_csv_writer.writerow(['ERA5 stationary GEV params'])
    era_csv_writer.writerow(list(obs_stat['mle_names']))
    era_csv_writer.writerow(list(obs_stat['mle']))
    era_csv_writer.writerow(['ERA5 stationary return period', str(np.around(orig_stat_rp,1))])
    era_csv.close()

    obs_gev_data={'gev_param_names' : obs_non_stat['mle_names'],
                  'gev_param_values': obs_non_stat['mle'],
                  'ana_year_value': ana_year_value, 
                  'ana_gsat_value': ana_gsat, 
                  'ana_year_RP': orig_nonstat_rp,
                  'ana_year_RP_CI': rp_perc,
                  'ano_obs_cb': ano_obs_cb,
                  'gsat_obs_cb': gsat_obs_cb}

    return ano_obs_cb, obs_gev_data


def create_gev_plot(data_dic, mixns, fit_param_apr, uncert_dic, obs_gev_data, cfg):

    ana_year = obs_gev_data['ana_year_value']
    ana_gsat = obs_gev_data['ana_gsat_value']

    era_shape = obs_gev_data['gev_param_values'][3] ; era_scale = obs_gev_data['gev_param_values'][2]
    era_loc = obs_gev_data['gev_param_values'][0] + obs_gev_data['gev_param_values'][1] * ana_gsat

    x_gev = uncert_dic.pop('x_gev')
    border = [np.floor(mixns['min']*1.5), np.ceil(mixns['max']*1.5)]

    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    risk_csv = open(os.path.join(cfg['work_dir'], 'analytics_gev_risk_data_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    risk_csv_writer = csv.writer(risk_csv, delimiter=',')
    risk_head_row = ['model','prob', 'RP', 'RP5', 'RP95']
    risk_csv_writer.writerow(risk_head_row)

    csv_file = open(os.path.join(cfg['work_dir'], 'analytics_gev_parameters_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['model', 'shape', 'shape_min', 'shape_5', 'shape_95', 'shape_max',
                'loc', 'loc_min', 'loc_5', 'loc_95', 'loc_max',
                'scale', 'scale_min', 'scale_5', 'scale_95', 'scale_max']
    gevs_csv_writer.writerow(head_row)

    for model in data_dic.keys(): 
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs = axs.flatten()
        fig.set_size_inches(12., 7.)
        ax_hist = axs[0]; ax_surv = axs[1] 
        model_row = [model]
        risk_model_row = [model]
        ens_var_cubelist = data_dic[model]['var_data']
        gsat_cubelist = data_dic[model]['gsat_data']
        distrib_var_data = []
        distrib_gsat_data = []
        weights = []
        for nvar, cube in enumerate(ens_var_cubelist):
            if (model == 'Multi-Model-Mean')&(cfg['model_weighting']):
                cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
            else:
                cube_weight = 1
            for n_p, point in enumerate(cube.data):
                distrib_var_data.append(np.around(point, 1))
                distrib_gsat_data.append(np.around(gsat_cubelist[nvar].data[n_p], 2))
                weights.append(cube_weight/len(cube.data))
        distrib_var_data = np.asarray(distrib_var_data)
        distrib_gsat_data = np.asarray(distrib_gsat_data)
        weights = np.asarray(weights)
        if cfg['model_weighting']:
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
            upd_var_distr_data = list()
            upd_gsat_distr_data = list()
            new_weights = list()
            for n_dp, distrib_point in enumerate(distrib_var_data): 
                for f in range(factors[n_dp]):
                    upd_var_distr_data.append(distrib_point)
                    upd_gsat_distr_data.append(distrib_gsat_data[n_dp])
                    new_weights.append(weights[n_dp]/factors[n_dp])
            distrib_var_data = np.asarray(upd_var_distr_data)
            distrib_gsat_data = np.asarray(upd_gsat_distr_data)
            weights = np.asarray(new_weights)
        w_distr_par = cex.fit_gev(distrib_var_data, distrib_gsat_data,  locationFun=1, returnValue=ana_year, getParams=True)
        try:
            w_shape = w_distr_par['mle'][3]; w_scale = w_distr_par['mle'][2] 
            w_loc = w_distr_par['mle'][0] + w_distr_par['mle'][1]*ana_gsat
            w_rp = 1/gev.sf(ana_year, -1*w_shape, loc=w_loc, scale=w_scale)
        except: 
            w_shape = np.nan ; w_scale = np.nan ; w_loc = np.nan; w_rp = np.nan
        w_survival = gev.sf(x_gev, -1*w_shape, loc=w_loc, scale=w_scale)
        era_surv = gev.sf(x_gev, -1*era_shape, loc=era_loc, scale=era_scale)
        model_row.extend([w_shape, uncert_dic[model]['shape_min'], uncert_dic[model]['shape_5']])
        model_row.extend([uncert_dic[model]['shape_95'], uncert_dic[model]['shape_max']])
        model_row.extend([w_loc, uncert_dic[model]['loc_min'], uncert_dic[model]['loc_5']])
        model_row.extend([uncert_dic[model]['loc_95'], uncert_dic[model]['loc_max']])            
        model_row.extend([w_scale, uncert_dic[model]['scale_min'], uncert_dic[model]['scale_5']])
        model_row.extend([uncert_dic[model]['scale_95'], uncert_dic[model]['scale_max']])            
        n_bins = np.arange(border[0], border[1]+0.1, 1)
        ax_hist.hist(distrib_var_data, bins=n_bins, edgecolor=col_mod,
                facecolor = col_mod, alpha=0.3, label=cfg['name_all'] , density=True, weights=weights, zorder = 5)
        ax_hist.hist(obs_gev_data['ano_obs_cb'].data, bins=n_bins, edgecolor=col_obs,
                facecolor = col_obs, alpha=0.3, label='ERA5 (1940-2022)' , density=True, zorder = 2)
        sf_perc_5 = uncert_dic[model]['sf_5th_perc']
        sf_perc_95 = uncert_dic[model]['sf_95th_perc']
        ax_surv.fill_between(x_gev, 1/sf_perc_5, 1/sf_perc_95, color=col_mod, alpha=0.3, linewidth=0, zorder=4)
        risk_model_row.extend([w_rp, np.nanpercentile(uncert_dic[model]['return_periods_all'],5), np.nanpercentile(uncert_dic[model]['return_periods_all'],95)])
        ax_surv.plot(x_gev, 1/w_survival, color=col_mod, zorder=5)
        ax_surv.plot(x_gev, 1/era_surv, color=col_obs, zorder=3)
        gevs_csv_writer.writerow(model_row)
        risk_csv_writer.writerow(risk_model_row)

        ylims = ax_hist.get_ylim()
        ax_hist.set_ylim(*ylims)

        ax_hist.text(border[0]/1.22, ylims[1]*0.65,'  Number of\nrealisations ' +str(len(ens_var_cubelist)), fontsize='large')
        ax_hist.vlines(ana_year, *ylims, color = 'indianred', linestyle = 'solid', lw=1.5, zorder=1, label = 'ERA5 ('+str(cfg['analysis_year'])+')')

        ax_surv.vlines(ana_year, 0.1, obs_gev_data['ana_year_RP'], linestyle = 'solid', color='indianred', zorder=2,  label = 'ERA5 ('+str(cfg['analysis_year'])+')')
        ax_surv.hlines(obs_gev_data['ana_year_RP'], x_gev[0], ana_year,linestyle = 'solid', color='indianred', zorder=2)

        ax_hist.legend(loc=2, fancybox=False, frameon=False)
        ax_hist.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_ylim(1,10000)
        ax_surv.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_yscale('log')
        ax_surv.grid(color='silver', axis='both', alpha=0.5)
        ax_hist.set_title('Probability density function')
        ax_surv.set_title('Return period')
        ax_surv.set_ylabel('years')
        ax_surv.set_xlabel(cfg['ax_var_label'] + ' anomaly, C')
        ax_hist.set_xlabel(cfg['ax_var_label'] + ' anomaly, C')
        ax_hist.set_ylabel('Number density')

        if model == 'Multi-Model-Mean':
            fig.suptitle(cfg['title_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+str(len(data_dic.keys()) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig.suptitle(cfg['title_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' calculated from '+model, fontsize = 'x-large')
        fig.set_dpi(250)

        plt.tight_layout()
        
        fig.savefig(os.path.join(cfg['plot_dir'], 'figure_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'_extremes_' +model + diagtools.get_image_format(cfg)))

    csv_file.close()
    risk_csv.close()

    return

def calculate_stds(data_dic, obs_gev_data, cfg):   

    era_var = obs_gev_data['ano_obs_cb'].data.std()
    
    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    fig_stds, ax_stds = plt.subplots(nrows=1, ncols=1)

    fig_stds.set_size_inches(8., 9.)
    fig_stds.set_dpi(200)

    y_ticks = np.arange(0, len(data_dic.keys()))
    y_labs = np.zeros(len(data_dic.keys()), dtype='<U30')

    for nm, model in enumerate(data_dic.keys()):
        if model != 'Multi-Model-Mean':
            for i in range(0, len(data_dic[model]['var_data'])):
                ax_stds.scatter(data_dic[model]['var_data'][i].data.std(), nm+1, s=50, marker='o', c=col_mod, zorder=2)
            y_labs[nm+1] = model
        else:
            wghts = list(); wdata = list()
            for i in range(0, len(data_dic[model]['var_data'])):
                wghts.append(np.full(data_dic[model]['var_data'][i].shape, data_dic[model]['var_data'][i].attributes['ensemble_weight']*data_dic[model]['var_data'][i].attributes['reverse_dtsts_n']))
                wdata.append(data_dic[model]['var_data'][i].data)
            wghts = np.asarray(wghts).flatten(); wdata = np.asarray(wdata).flatten()
            if cfg['model_weighting']:
                mmm = np.average(wdata, weights=wghts)
                mmm_std = np.sqrt(np.average((wdata - mmm)**2, weights=wghts))
            else: 
                mmm_std = wdata.std()
            ax_stds.scatter(mmm_std, 0, c = col_mod, s=70, marker='s', zorder=2)
            y_labs[0] = 'CMIP6'
    
    ax_stds.axvline(era_var, -1, len(data_dic.keys()) + 1, c=col_obs, zorder=1)
    ax_stds.set_ylim(len(data_dic.keys()) -0.8, -0.2)

    ax_stds.set_yticks(y_ticks, labels=y_labs)
    ax_stds.grid(which='both', c='silver')

    ax_stds.set_xlabel('StD of '+cfg['ax_var_label'] + ' anomalies, C')

    fig_stds.suptitle('Standard deviations (StD) of '+cfg['title_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1]))

    plt.tight_layout()

    fig_stds.savefig(os.path.join(cfg['plot_dir'], 'stds_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.'+cfg['output_file_type']))           


    return

def create_timeseries(data_dic, mixns, obs_gev_data, cfg):

    b_max = np.ceil(mixns['max']*1.05)
    b_min = np.floor(mixns['min']*1.05)

    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    t_s = np.arange(cfg['year_span'][0], cfg['year_span'][1]+1)

    for dataset in data_dic.keys(): 
        fig_ts, ax_ts = plt.subplots(nrows=2, ncols=1)
        ax_ts = ax_ts.flatten()
        fig_ts.set_size_inches(10., 8.)
        fig_ts.set_dpi(200)

        model_var_data = list()
        model_gsat_data = list()
        weights = list()

        for i in range(0,len(data_dic[dataset]['var_data'])):
            model_var_data.append(data_dic[dataset]['var_data'][i].data)
            model_gsat_data.append(data_dic[dataset]['gsat_data'][i].data)
            weights.append(data_dic[dataset]['var_data'][i].attributes['ensemble_weight']*
                            data_dic[dataset]['var_data'][i].attributes['reverse_dtsts_n'])
        weights = np.array(weights)
        model_var_data = np.array(model_var_data)
        model_gsat_data = np.array(model_gsat_data)
        if (dataset == 'Multi-Model-Mean')&(cfg['model_weighting']):
            mean_var_arr = np.average(model_var_data, axis=0, weights=weights)
            mean_gsat_arr = np.average(model_gsat_data, axis=0, weights=weights)
            # so far std, maybe do percentiles
            std_var_arr = np.sqrt(np.average((model_var_data - mean_var_arr)**2, axis=0, weights=weights))
            std_gsat_arr = np.sqrt(np.average((model_gsat_data - mean_gsat_arr)**2, axis=0, weights=weights))
        else:
            mean_var_arr = np.average(model_var_data, axis=0)
            min_var_arr = np.min(model_var_data, axis=0)
            max_var_arr = np.max(model_var_data, axis=0)
            mean_gsat_arr = np.average(model_gsat_data, axis=0)
            perc_gsat_arr = np.percentile(model_gsat_data, [5,95], axis=0)

        ax_ts[0].text(0.47, 0.95, cfg['ax_var_label'], transform=ax_ts[0].transAxes)
        ax_ts[1].text(0.45, 0.95, 'Smoothed GSAT', transform=ax_ts[1].transAxes)
        ax_ts[0].plot(t_s, obs_gev_data['ano_obs_cb'].data, c=col_obs, zorder=5, label='ERA5')
        ax_ts[0].plot(t_s, mean_var_arr, c=col_mod, zorder=3, label=dataset)
        ax_ts[1].plot(t_s, obs_gev_data['gsat_obs_cb'].data, c=col_obs, zorder=5, label='ERA5')
        ax_ts[1].plot(t_s, mean_gsat_arr, c=col_mod, zorder=3, label=dataset)
        if len(data_dic[dataset]['var_data'])>1:
            # clean this noncense before submitting! 
            try: 
                ax_ts[0].fill_between(t_s, min_var_arr, max_var_arr, color=col_mod, lw=0, alpha=0.25)
                ax_ts[1].fill_between(t_s, perc_gsat_arr[0], perc_gsat_arr[1], color=col_mod, lw=0, alpha=0.25)
            except:
                ax_ts[0].fill_between(t_s, mean_var_arr+std_var_arr, mean_var_arr-std_var_arr, color=col_mod, lw=0, alpha=0.25)
                ax_ts[1].fill_between(t_s, mean_gsat_arr+std_gsat_arr, mean_gsat_arr-std_gsat_arr, color=col_mod, lw=0, alpha=0.25)
        [a.plot([t_s[0]-0.5, t_s[-1]+0.5], [0,0], c='grey') for a in ax_ts]
        [a.set_xlim(t_s[0]-0.5, t_s[-1]+0.5) for a in ax_ts]
        ax_ts[0].set_ylim(b_min, b_max); ax_ts[1].set_ylim(-1, 2)  # so far fixed, revise
        [a.legend(loc=0, ncols=2, fancybox=False, frameon=False) for a in ax_ts]
        ax_ts[0].set_ylabel(cfg['ax_var_label'] + ' anomaly, C')
        ax_ts[1].set_ylabel('GSAT anomaly, C')
        ax_ts[1].set_xlabel('year')
        if dataset == 'Multi-Model-Mean':
            fig_ts.suptitle('Anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' from '+str(len(data_dic.keys()) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig_ts.suptitle('Anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ '  from '+dataset, fontsize = 'x-large')

        plt.tight_layout()

        fig_ts.savefig(os.path.join(cfg['plot_dir'], 'ts_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'_'+dataset+'.'+cfg['output_file_type']))

    return

def obtain_smoothed_gsat(gsat_group, dataset, ensemble, cfg):

    gsat_cb=iris.load_cube(select_metadata(gsat_group, dataset=dataset, ensemble=ensemble)[0]['filename'])
    gsat_df = pd.DataFrame(gsat_cb.data, columns=['gsat'])

    # smoothing gsat
    kernel_size = cfg['smooth_gsat_years']
    gsat_smooth_df = gsat_df.rolling(kernel_size, min_periods=1).mean()
    gsat_smooth_arr = gsat_smooth_df['gsat'].to_numpy()
    smoothed_cube = gsat_cb
    smoothed_cube.data = gsat_smooth_arr

    return smoothed_cube


def main(cfg):

    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)

    ano_obs_cb, obs_gev_data = obtain_obs_info(groups, cfg)

    # change later, so far to identify the group name
    group = [k for k in groups.keys() if 'txnx' in k][0]

    plotting_dic = {}
    fit_param_apr = {}
    mixns = {}

    mins = list(); maxs = list()
    group_data = groups[group]
    group_gsat_data = groups['gsat_'+group.split('_')[-1]]
    datasets = group_metadata(group_data, 'dataset')
    ens_var_cubelist = iris.cube.CubeList()
    ens_gsat_cubelist = iris.cube.CubeList()
    for dataset in datasets.keys():
        filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
        n_real = len(filepaths)
        mod_var_cubelist = iris.cube.CubeList()
        mod_gsat_cubelist = iris.cube.CubeList()
        for filepath in filepaths:
            mod_cb = iris.load_cube(filepath)
            file_metadata = select_metadata(datasets[dataset], filename = filepath)
            ens = file_metadata[0]['ensemble']
            gsat_cb = obtain_smoothed_gsat(group_gsat_data, dataset, ens, cfg)
            # adding weights to the data cubes 
            mod_cb.attributes['ensemble_weight'] = 1 / n_real
            mod_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
            gsat_cb.attributes['ensemble_weight'] = 1 / n_real
            gsat_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
            ens_var_cubelist.append(mod_cb); mod_var_cubelist.append(mod_cb)
            ens_gsat_cubelist.append(gsat_cb); mod_gsat_cubelist.append(gsat_cb)
            mins.append(mod_cb.collapsed('time', iris.analysis.MIN).data)
            maxs.append(mod_cb.collapsed('time', iris.analysis.MAX).data)
        plotting_dic[dataset] = {'var_data': mod_var_cubelist}
        plotting_dic[dataset]['gsat_data'] = mod_gsat_cubelist
        plotting_dic[dataset]['uncert'] = bootstrap_gev(mod_var_cubelist, obs_gev_data, mod_gsat_cubelist)
    mixns['max'] = np.asarray(maxs).max()
    mixns['min'] = np.asarray(mins).min()
    plotting_dic['Multi-Model-Mean'] = {'var_data' : ens_var_cubelist, 'gsat_data': ens_gsat_cubelist}
    plotting_dic['Multi-Model-Mean']['uncert'] = bootstrap_gev(plotting_dic, obs_gev_data) 
    fit_param_apr = {'loc': np.around(plotting_dic['Multi-Model-Mean']['uncert']['loc'].mean(),3),
                        'scale': np.around(plotting_dic['Multi-Model-Mean']['uncert']['scale'].mean(),3),
                        'shape': np.around(plotting_dic['Multi-Model-Mean']['uncert']['shape'].mean(),3)}

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    uncert_band = calculate_uncert_band(plotting_dic, mixns, cfg)

    create_gev_plot(plotting_dic, mixns, fit_param_apr, uncert_band, obs_gev_data, cfg)

    create_timeseries(plotting_dic, mixns, obs_gev_data, cfg)

    calculate_stds(plotting_dic, obs_gev_data, cfg)

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
