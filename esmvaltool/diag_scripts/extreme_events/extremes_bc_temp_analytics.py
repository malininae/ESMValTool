import csv
import esmvalcore.preprocessor as eprep
import iris
from iris.util import equalise_attributes
import iris.plot as iplt
import cf_units
import cftime 
import cartopy.crs as ccrs 
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r as gumbel
from scipy.stats import kstest, cramervonmises

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata, get_diagnostic_filename, save_data, ProvenanceLogger
import esmvaltool.diag_scripts.shared.plot as eplot
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.extreme_events.extremes_bc_temperature import bootstrap_gev
from esmvaltool.diag_scripts.shared import ProvenanceLogger

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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
            gev_pdf = gev.pdf(x_gev, mod_gevs['shape'][i],mod_gevs['loc'][i], mod_gevs['scale'][i])
            gev_sf = gev.sf(x_gev, mod_gevs['shape'][i],mod_gevs['loc'][i], mod_gevs['scale'][i])
            all_pdfs[:, i] = gev_pdf
            all_sfs[:, i] = gev_sf
            uncert_band[model] = {'pdf_5th_perc' : np.percentile(all_pdfs, 5, axis = 1), 
                                'pdf_95th_perc': np.percentile(all_pdfs, 95, axis = 1),
                                'sf_5th_perc' : np.percentile(all_sfs, 5, axis = 1), 
                                'sf_95th_perc' : np.percentile(all_sfs, 95, axis = 1),
                                'sf_data_full' : all_sfs,
                                'loc_min': mod_gevs['loc'].min(), 'loc_max': mod_gevs['loc'].max(),
                                'loc_5': np.percentile(mod_gevs['loc'], 5, interpolation='nearest'),
                                'loc_95': np.percentile(mod_gevs['loc'], 95, interpolation='nearest'),
                                'scale_min': mod_gevs['scale'].min(), 'scale_max': mod_gevs['scale'].max(),
                                'scale_5': np.percentile(mod_gevs['scale'], 5, interpolation='nearest'),
                                'scale_95': np.percentile(mod_gevs['scale'], 95, interpolation='nearest'),
                                'shape_min': mod_gevs['shape'].min(), 'shape_max': mod_gevs['shape'].max(),
                                'shape_5': np.percentile(mod_gevs['shape'], 5, interpolation='nearest'),
                                'shape_95': np.percentile(mod_gevs['shape'], 95, interpolation='nearest')}

    return uncert_band

def get_ref_params(data_cb, ana_year, cfg, distrib = 'gev'):

    if distrib.lower() == 'gev':
        era_gev_params = gev.fit(data_cb.data)
    elif distrib.lower() == 'gumbel':
        era_gev_params = gumbel.fit(data_cb.data)
    
    era_ks = kstest(data_cb.data, gev(*era_gev_params).cdf)
    era_cvm = cramervonmises(data_cb.data, gev(*era_gev_params).cdf)
    
    orig_rp = np.around(1/gev.sf(ana_year, *era_gev_params), 1)
    bootstrap_rps = list()
    rng = np.random.default_rng(501)

    for i in range(1000): 
        for_fit = rng.choice(data_cb.data, size=len(data_cb.data), replace=True)
        for_fit = np.append(for_fit, ana_year)
        temp_gev = gev.fit(for_fit)
        temp_rp = np.around(1/gev.sf(ana_year, *temp_gev), 1)
        bootstrap_rps.append(temp_rp)
    
    bootstrap_rps = np.asarray(bootstrap_rps)

    rp_perc = np.percentile(bootstrap_rps, [5,10,50,90,95]).round(1)

    era_csv = open(os.path.join(cfg['work_dir'], distrib.lower()+'_era_data_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    era_csv_writer = csv.writer(era_csv, delimiter=',')
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA value '+str(ana_year)])
    era_csv_writer.writerow(['ERA ' +distrib +' params'])
    if distrib.lower() == 'gev':
        era_csv_writer.writerow(['shape', 'loc', 'scale'])
    elif distrib.lower() == 'gumbel':
        era_csv_writer.writerow(['loc', 'scale'])
    era_csv_writer.writerow(era_gev_params)
    era_csv_writer.writerow(['KS test params: statistic', 'pvalue'])
    era_csv_writer.writerow([era_ks.statistic, era_ks.pvalue])
    era_csv_writer.writerow(['CvM test params: statistic', 'pvalue'])
    era_csv_writer.writerow([era_cvm.statistic, era_cvm.pvalue])
    era_csv_writer.writerow([str(cfg['analysis_year'])+' ERA return period', str(np.around(orig_rp,1))])
    era_csv_writer.writerow(['Bootstrapped uncertanties on ERA return period'])
    era_csv_writer.writerow(['5_perc', '10_perc', '50_perc', '90_perc', '95_perc'])
    era_csv_writer.writerow(rp_perc)
    era_csv.close()

    return era_gev_params


def create_gev_plot(data_dic, mixns, fit_param_apr, uncert_dic, ano_shp_cb, cfg):

    ana_year_const = iris.Constraint(time = lambda cell: cell.point.year == cfg['analysis_year'])
    ana_year =  ano_shp_cb.extract(ana_year_const).data

    era_gevs = get_ref_params(ano_shp_cb, ana_year, cfg)

    x_gev = uncert_dic.pop('x_gev')
    border = [np.floor(mixns['min']*1.5), np.ceil(mixns['max']*1.5)]

    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    risk_csv = open(os.path.join(cfg['work_dir'], 'gev_risk_data_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    risk_csv_writer = csv.writer(risk_csv, delimiter=',')
    risk_head_row = ['model','prob', 'RP', 'RP5', 'RP95', 'intens', 'intens5', 'intens95']
    risk_csv_writer.writerow(risk_head_row)

    csv_file = open(os.path.join(cfg['work_dir'], 'gev_parameters'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'.csv'), 'w', newline='')
    gevs_csv_writer = csv.writer(csv_file, delimiter=',')
    head_row = ['model', 'shape', 'shape_min', 'shape_5', 'shape_95', 'shape_max',
                'loc', 'loc_min', 'loc_5', 'loc_95', 'loc_max',
                'scale', 'scale_min', 'scale_5', 'scale_95', 'scale_max', 
                'ks_mod_stat', 'ks_mod_p_value', 'cvm_mod_stat', 'cvm_mod_p_value',
                'ks_obs_stat', 'ks_obs_p_value', 'cvm_obs_stat', 'cvm_obs_p_value']
    gevs_csv_writer.writerow(head_row)

    quantile_measures = np.arange(0, 1.01, 0.01); quantile_measures[0] = 0.001

    for model in data_dic.keys(): 
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(12., 8.)
        gs = fig.add_gridspec(nrows=2, ncols=6)
        ax_hist = fig.add_subplot(gs[:, :-2])
        ax_qq= fig.add_subplot(gs[0, -2:])
        ax_surv= fig.add_subplot(gs[1, -2:])
        model_row = [model]
        risk_model_row = [model]
        ens_cubelist = data_dic[model]['data']
        distrib_data = []
        weights = []
        for cube in ens_cubelist:
            if (model == 'Multi-Model-Mean')&(cfg['model_weighting']):
                cube_weight = cube.attributes['ensemble_weight']*cube.attributes['reverse_dtsts_n']
            else:
                cube_weight = 1
            for point in cube.data:
                distrib_data.append(np.around(point, 1))
                weights.append(cube_weight/len(cube.data))
        distrib_data = np.asarray(distrib_data)
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
            upd_distr_data = list()
            new_weights = list()
            for n_dp, distrib_point in enumerate(distrib_data): 
                for f in range(factors[n_dp]):
                    upd_distr_data.append(distrib_point)
                    new_weights.append(weights[n_dp]/factors[n_dp])
            distrib_data = np.asarray(upd_distr_data)
            weights = np.asarray(new_weights)
        w_distr_par = gev.fit(distrib_data, loc=fit_param_apr['loc'], scale=fit_param_apr['scale'], method='MLE')
        w_pdf = gev.pdf(x_gev, *w_distr_par)
        w_survival = gev.sf(x_gev, *w_distr_par)
        theor_quants = gev(*w_distr_par).ppf(quantile_measures)
        era_surv = gev.sf(x_gev, *era_gevs)
        ks_mod_res = kstest(distrib_data, gev(*w_distr_par).cdf)
        cvm_mod_res = cramervonmises(distrib_data, gev(*w_distr_par).cdf)
        ks_obs_res = kstest(distrib_data, gev(*era_gevs).cdf)
        cvm_obs_res = cramervonmises(distrib_data, gev(*era_gevs).cdf)
        event_idx = np.argmin(np.abs(x_gev - ana_year))
        era_event_prob = era_surv[event_idx]
        model_row.extend([w_distr_par[-3], uncert_dic[model]['shape_min'], uncert_dic[model]['shape_5']])
        model_row.extend([uncert_dic[model]['shape_95'], uncert_dic[model]['shape_max']])
        model_row.extend([w_distr_par[-2], uncert_dic[model]['loc_min'], uncert_dic[model]['loc_5']])
        model_row.extend([uncert_dic[model]['loc_95'], uncert_dic[model]['loc_max']])            
        model_row.extend([w_distr_par[-1], uncert_dic[model]['scale_min'], uncert_dic[model]['scale_5']])
        model_row.extend([uncert_dic[model]['scale_95'], uncert_dic[model]['scale_max']])            
        model_row.extend([ks_mod_res.statistic, ks_mod_res.pvalue, cvm_mod_res.statistic, cvm_mod_res.pvalue])
        model_row.extend([ks_obs_res.statistic, ks_obs_res.pvalue, cvm_obs_res.statistic, cvm_obs_res.pvalue])
        n_bins = np.arange(border[0], border[1]+0.1, 1)
        ax_hist.hist(distrib_data, bins=n_bins, edgecolor=col_mod,
                facecolor = col_mod, alpha=0.3, label=cfg['name_all'] , density=True, weights=weights, zorder = 5)
        ax_hist.plot(x_gev, w_pdf, c = col_mod, ls = 'solid', label = 'GEV fit '+ cfg['name_all'], zorder=6)
        ax_hist.hist(ano_shp_cb.data, bins=n_bins, edgecolor=col_obs,
                facecolor = col_obs, alpha=0.3, label='ERA5 (1950-2022)' , density=True, zorder = 2)
        ax_hist.plot(x_gev, gev.pdf(x_gev, *era_gevs), c = col_obs, ls = 'solid', label = 'GEV fit ERA5', zorder=3)
        pdf_perc_5 = uncert_dic[model]['pdf_5th_perc']
        pdf_perc_95 = uncert_dic[model]['pdf_95th_perc']
        sf_perc_5 = uncert_dic[model]['sf_5th_perc']
        sf_perc_95 = uncert_dic[model]['sf_95th_perc']
        ax_hist.fill_between(x_gev, pdf_perc_5, pdf_perc_95, color=col_mod, alpha=0.3, linewidth=0, zorder=4)
        ax_surv.fill_between(x_gev, 1/sf_perc_5, 1/sf_perc_95, color=col_mod, alpha=0.3, linewidth=0, zorder=4)
        intens = x_gev[np.argmin(np.abs(w_survival-era_event_prob))]
        intens_95 = x_gev[np.argmin(np.abs(sf_perc_95-era_event_prob))]
        intens_5 = x_gev[np.argmin(np.abs(sf_perc_5-era_event_prob))]
        risk_model_row.extend([w_survival[event_idx], 1/w_survival[event_idx], 1/sf_perc_5[event_idx], 1/sf_perc_95[event_idx], intens, intens_5, intens_95])
        pract_quants = np.quantile(distrib_data, quantile_measures)
        pract_era_quants = np.quantile(ano_shp_cb.data, quantile_measures)
        ax_qq.scatter(theor_quants, pract_quants, edgecolors=col_mod, marker='o', facecolors='None', lw=0.75, label=cfg['name_all'], zorder=5)
        ax_surv.plot(x_gev, 1/w_survival, color=col_mod, zorder=5)
        ax_qq.scatter(theor_quants, pract_era_quants, edgecolors=col_obs, marker='o', facecolors='None', lw=0.75, label='ERA5', zorder=3)
        ax_surv.plot(x_gev, 1/gev.sf(x_gev, *era_gevs), color=col_obs, zorder=3)
        gevs_csv_writer.writerow(model_row)
        risk_csv_writer.writerow(risk_model_row)

        ylims = ax_hist.get_ylim()
        ax_hist.set_ylim(*ylims)

        ax_hist.text(border[0]/1.22, ylims[1]*0.65,'  Number of\nrealisations ' +str(len(ens_cubelist)), fontsize='large')
        ax_hist.vlines(ana_year, *ylims, color = 'indianred', linestyle = 'solid', lw=1.5, zorder=1, label = 'ERA5 ('+str(cfg['analysis_year'])+')')

        ax_surv.vlines(ana_year, 0.1, 1/era_event_prob, linestyle = 'solid', color='indianred', zorder=2,  label = 'ERA5 ('+str(cfg['analysis_year'])+')')
        ax_surv.hlines(1/era_event_prob, x_gev[0], ana_year,linestyle = 'solid', color='indianred', zorder=2)

        ax_hist.legend(loc=2, fancybox=False, frameon=False)
        ax_qq.legend(loc=2, fancybox=False, frameon=False, handletextpad=0.01)
        ax_qq.plot(border, border, c='tab:grey', zorder=1)
        ax_qq.grid(color='silver', axis='both', alpha=0.5)
        ax_hist.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_qq.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_qq.set_ylim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_ylim(1,10000)
        ax_surv.set_xlim(border[0]/1.2, border[1]/1.2)
        ax_surv.set_yscale('log')
        ax_surv.grid(color='silver', axis='both', alpha=0.5)
        ax_hist.set_title('Probability density function')
        ax_qq.set_title('Quality assessment')
        ax_qq.set_ylabel('Data quantile')
        ax_qq.set_xlabel('GEV quantile')
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

def calculate_stds(data_dic, ano_shp_cb, cfg):   

    era_var = ano_shp_cb.data.std()
    
    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    fig_stds, ax_stds = plt.subplots(nrows=1, ncols=1)

    fig_stds.set_size_inches(8., 9.)
    fig_stds.set_dpi(200)

    y_ticks = np.arange(0, len(data_dic.keys()))
    y_labs = np.zeros(len(data_dic.keys()), dtype='<U30')

    for nm, model in enumerate(data_dic.keys()):
        if model != 'Multi-Model-Mean':
            for i in range(0, len(data_dic[model]['data'])):
                ax_stds.scatter(data_dic[model]['data'][i].data.std(), nm+1, s=50, marker='o', c=col_mod, zorder=2)
            y_labs[nm+1] = model
        else:
            wghts = list(); wdata = list()
            for i in range(0, len(data_dic[model]['data'])):
                wghts.append(np.full(data_dic[model]['data'][i].shape, data_dic[model]['data'][i].attributes['ensemble_weight']*data_dic[model]['data'][i].attributes['reverse_dtsts_n']))
                wdata.append(data_dic[model]['data'][i].data)
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

def create_timeseries(data_dic, mixns, ano_shp_cb, cfg):

    b_max = np.ceil(mixns['max']*1.05)
    b_min = np.floor(mixns['min']*1.05)

    col_mod = (25/255, 14/255, 79/255)
    col_obs = 'indianred'

    t_s = np.arange(cfg['year_span'][0], cfg['year_span'][1]+1)

    for dataset in data_dic.keys(): 
        fig_ts, ax_ts = plt.subplots(nrows=1, ncols=1)
        fig_ts.set_size_inches(10., 5.)
        fig_ts.set_dpi(200)

        model_data = list()
        weights = list()

        for i in range(0,len(data_dic[dataset]['data'])):
            model_data.append(data_dic[dataset]['data'][i].data)
            weights.append(data_dic[dataset]['data'][i].attributes['ensemble_weight']*
                            data_dic[dataset]['data'][i].attributes['reverse_dtsts_n'])
        weights = np.array(weights)
        model_data = np.array(model_data)
        if (dataset == 'Multi-Model-Mean')&(cfg['model_weighting']):
            mean_arr = np.average(model_data, axis=0, weights=weights)
            std_arr = np.sqrt(np.average((model_data - mean_arr)**2, axis=0, weights=weights))
        else:
            mean_arr = np.average(model_data, axis=0)
            std_arr = np.std(model_data, axis=0)

        ax_ts.plot(t_s, ano_shp_cb.data, c=col_obs, zorder=5, label='ERA5')
        ax_ts.plot(t_s, mean_arr, c=col_mod, zorder=3, label=dataset)
        if len(data_dic[dataset]['data'])>1: 
            ax_ts.fill_between(t_s, mean_arr+std_arr, mean_arr-std_arr, color=col_mod, lw=0, alpha=0.25)
        ax_ts.plot([t_s[0]-0.5, t_s[-1]+0.5], [0,0], c='grey')        
        ax_ts.set_xlim(t_s[0]-0.5, t_s[-1]+0.5)
        ax_ts.set_ylim(b_min, b_max)
        ax_ts.legend(loc=0, ncols=2, fancybox=False, frameon=False)
        ax_ts.set_ylabel(cfg['ax_var_label'] + ' anomaly, C')
        ax_ts.set_xlabel('year')
        if dataset == 'Multi-Model-Mean':
            fig_ts.suptitle(cfg['title_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ ' from '+str(len(data_dic.keys()) - 1) +' CMIP6 models' , fontsize = 'x-large')
        else:
            fig_ts.suptitle(cfg['title_var_label']+' anomalies in '+cfg['region']+' relative to '+ str(cfg['reference_period'][0]) \
            + '-' + str(cfg['reference_period'][1])+ '  from '+dataset, fontsize = 'x-large')

        plt.tight_layout()

        fig_ts.savefig(os.path.join(cfg['plot_dir'], 'ts_'+cfg['region'].lower()+'_'+cfg['ax_var_label'].lower()+'_'+dataset+'.'+cfg['output_file_type']))

    return


def main(cfg):

    input_data = cfg['input_data']

    groups = group_metadata(input_data.values(), 'variable_group', sort=True)
    obs_info = groups.pop('obs')
    ano_shp_cb = iris.load_cube(obs_info[0]['filename'])

    groups_l = list(groups.keys())

    distrib_fit= cfg['fit_distribution']

    plotting_dic = {}
    fit_param_apr = {}
    mixns = {}

    for group in groups_l:
        mins = list(); maxs = list()
        plotting_dic[group] = {}; mixns[group] = {}
        group_data = groups[group]
        datasets = group_metadata(group_data, 'dataset')
        ens_cubelist = iris.cube.CubeList()
        for dataset in datasets.keys():
            filepaths = list(group_metadata(datasets[dataset], 'filename').keys())
            n_real = len(filepaths)
            mod_cubelist = iris.cube.CubeList()
            for filepath in filepaths:
                mod_cb = iris.load_cube(filepath)
                mod_cb.attributes['ensemble_weight'] = 1 / n_real
                mod_cb.attributes['reverse_dtsts_n'] = 1/ len(datasets)
                ens_cubelist.append(mod_cb)
                mod_cubelist.append(mod_cb)
                mins.append(mod_cb.collapsed('time', iris.analysis.MIN).data)
                maxs.append(mod_cb.collapsed('time', iris.analysis.MAX).data)
            plotting_dic[group][dataset] = {'data': mod_cubelist}
            plotting_dic[group][dataset]['uncert'] = bootstrap_gev(mod_cubelist, distrib=distrib_fit)
        mixns[group]['max'] = np.asarray(maxs).max()
        mixns[group]['min'] = np.asarray(mins).min()
        plotting_dic[group]['Multi-Model-Mean'] = {'data' : ens_cubelist}
        plotting_dic[group]['Multi-Model-Mean']['uncert'] = bootstrap_gev(plotting_dic[group], distrib=distrib_fit) 
        fit_param_apr[group] = {'loc': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['loc'].mean(),3),
                            'scale': np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['scale'].mean(),3)}
        if distrib_fit.lower() == 'gev':
            fit_param_apr[group]['shape'] = np.around(plotting_dic[group]['Multi-Model-Mean']['uncert']['shape'].mean(),3)

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    uncert_band = calculate_uncert_band(plotting_dic['txx_all'], mixns['txx_all'], cfg)

    create_gev_plot(plotting_dic['txx_all'], mixns['txx_all'], fit_param_apr['txx_all'], uncert_band, ano_shp_cb, cfg)

    create_timeseries(plotting_dic['txx_all'], mixns['txx_all'], ano_shp_cb, cfg)

    calculate_stds(plotting_dic['txx_all'], ano_shp_cb, cfg)

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
