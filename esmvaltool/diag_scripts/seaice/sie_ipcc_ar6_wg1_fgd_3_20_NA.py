# This is a script to create a figure 3.20 in Chapter 3 IPCC WGI AR6
# Authors: Elizaveta Malinina (elizaveta.malinina-rieger@canada.ca)
#          Seung-Ki Min, Yeon-Hee Kim, Nathan Gillett

import iris
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_entry_stat(cubelist):

    list_dic = []

    for cube in cubelist:
        if np.all(cube.data.mask) | (np.all(cube.data == 0)):
            mean = np.nan
            slope = np.nan
        else:
            x = np.arange(0, len(cube.coord('time').points))  # here we calculate the trend, orig code logic
            reg = stats.linregress(x, cube.data)
            mean = np.average(cube.data)
            slope = reg.slope
        # list of dictionaries. Not beautiful, but works.
        # It is not needed to know which realisation the data belongs to.
        list_dic.append({'mean': mean, 'lin_reg_slope': slope})

    return (list_dic)


def ens_average(ens_list_dic):

    if len(ens_list_dic) == 1:
        mod_mean = ens_list_dic[0]['mean']
        mod_dec_slope = ens_list_dic[0]['lin_reg_slope'] * 10
        mod_means = np.asarray([mod_mean]) ; mod_dec_slopes = np.asarray([mod_dec_slope])
    else:
        mod_means = np.asarray([entry['mean'] for entry in ens_list_dic])
        mod_dec_slopes = np.asarray([entry['lin_reg_slope']*10 for entry in ens_list_dic])
        mod_mean = np.nanmean(mod_means)
        mod_dec_slope = np.nanmean(mod_dec_slopes)

    return (mod_mean, mod_dec_slope, mod_means, mod_dec_slopes)


def model_stats(inp_dict):

    all_means = np.asarray(list())
    all_slopes = np.asarray(list())
    all_weights = np.asarray(list())

    for key in inp_dict.keys(): 
        inp_dict[key]['mean_std'] = np.std(inp_dict[key]['means_all'])
        inp_dict[key]['slope_std'] = np.std(inp_dict[key]['slopes_all'])
        all_means = np.concatenate([all_means,inp_dict[key]['means_all']])
        all_slopes = np.concatenate([all_slopes,inp_dict[key]['slopes_all']])
        all_weights = np.concatenate([all_weights,np.full(len(inp_dict[key]['means_all']),1/len(inp_dict[key]['means_all']))])

    all_weights = all_weights/len(all_weights)

    means = np.asarray([inp_dict[key]['mean'] for key in inp_dict.keys()])
    slopes = np.asarray([inp_dict[key]['dec_slope'] for key in inp_dict.keys()])

    # checks if we're not comparing numbers with nans
    mask = np.isfinite(means) & np.isfinite(slopes)

    reg = stats.linregress(means[mask], slopes[mask])

    # calculating p value

    tval = reg.slope / reg.stderr
    df = len(means[mask]) - 2
    pval = special.betainc(df / 2, 0.5, df / (df + tval ** 2))  # this particular calculation was adopted from orig code

    inp_dict['stat_params'] = {}
    inp_dict['stat_params']['slope_models'] = reg.slope
    inp_dict['stat_params']['mme_mean'] = np.average(means[mask])
    inp_dict['stat_params']['mme_slope'] = np.average(slopes[mask])
    inp_dict['stat_params']['mme_mean_std'] = np.sqrt(np.average((all_means - np.average(all_means, weights=all_weights))**2, weights=all_weights))
    inp_dict['stat_params']['mme_slope_std'] = np.sqrt(np.average((all_slopes - np.average(all_slopes, weights=all_weights))**2, weights=all_weights))
    inp_dict['stat_params']['intercept'] = reg.intercept
    inp_dict['stat_params']['p_val'] = pval
    inp_dict['stat_params']['corr_coef'] = stats.pearsonr(means[mask], slopes[mask])[0]

    return (inp_dict)


def make_panel(data_dict, nrow, ncol, idx, obs_dic, verb_month, hemisph, proj):

    obs_cbar = plt.cm.Greys_r

    if (hemisph == 'NH')|(hemisph == 'MH'):
        region = 'Arctic'
    elif hemisph == 'SH':
        region = 'Antarctic'
    else:   
        region = 'N.Atlantic'

    title = region + ' SIA in ' + verb_month

    ax = plt.subplot(nrow, ncol, idx)

    ax.set_title(title)
    stat_params = data_dict.pop('stat_params')
    obs_cmap_step = int(200 / len(obs_dic.keys()))
    xs = []

    obs_scat = list()
    for n_o, obs in enumerate(obs_dic.keys()):
        obs_scat_p = ax.scatter(obs_dic[obs]['mean'], obs_dic[obs]['dec_slope'],
                                label='OBS: '+obs.split('-')[0], s=200, marker="*", zorder=2,
                                c=obs_cbar(n_o * obs_cmap_step))
        obs_scat.append(obs_scat_p)

    mme_sty = eplot.get_dataset_style('MultiModelMean', proj.lower() +'_canesm.yml')
    mme_scat = ax.scatter(stat_params['mme_mean'], stat_params['mme_slope'], 
                s=100, marker=mme_sty['mark'], c=mme_sty['color'],
                label='Multi Model Mean', zorder=5)
    # adding legend
    ax.errorbar(-10, -10, xerr=2, yerr=2,marker='o', c='#757575',
               label='CMIP6 single models', ms=6)

    mod_obs = list()
    for n, model in enumerate(sorted(data_dict.keys())):
        sty = eplot.get_dataset_style(model, proj.lower() +'_canesm.yml')
        if 'CanESM' in model:
            mod_obs_p = ax.errorbar(data_dict[model]['mean'], data_dict[model]['dec_slope'], label = model, 
                                yerr=data_dict[model]['slope_std'],xerr=data_dict[model]['mean_std'], 
                                c=sty['color'], marker=sty['mark'], ms=7, zorder=3)
            # mod_obs_p = ax.scatter(data_dict[model]['mean'], data_dict[model]['dec_slope'],
            #         label=model, c=sty['color'], marker=sty['mark'], s=50, zorder=2)
            # ax.scatter(data_dict[model]['means_all'], data_dict[model]['slopes_all'], edgecolor='none',
            #            facecolor = sty['color'], linewidth=0, alpha=0.2, s=50, zorder=4)
        else:
            # mod_obs_p = ax.scatter(data_dict[model]['mean'], data_dict[model]['dec_slope'],
            #         edgecolor=sty['color'], facecolor='none',
            #         linewidths=2, marker=sty['mark'], s=50, zorder=2)
            mod_obs_p = ax.errorbar(data_dict[model]['mean'], data_dict[model]['dec_slope'],
                                    yerr=data_dict[model]['slope_std'],xerr=data_dict[model]['mean_std'], 
                                    c=sty['color'], marker=sty['mark'], ms=7, zorder=3)
        mod_obs.append(mod_obs_p)
        xs.append(data_dict[model]['mean'])

    # xs = np.asarray(xs)
    # lin, = ax.plot(xs, xs * stat_params['slope_models'] + stat_params['intercept'], c='k', zorder=1)

    if idx == 1:
        ax.set_ylabel(r'Trend(10$^6$ km$^2$/ decade)')
    ax.set_xlabel(r'Mean (10$^6$ km$^2$)')

    if hemisph == 'NH':
        ax.set_ylim(-1.5, 0)
        ax.set_xlim(2, 10)
        ax.set_yticks(np.arange(-1.5, 0.1, 0.3))
        y_text = -1.45
        x_text = 6.5
    elif hemisph == 'SH':
        ax.set_ylim(-0.9, 0.3)
        ax.set_xlim(-0.5, 7)
        ax.set_yticks(np.arange(-0.9, 0.4, 0.3))
        y_text = -0.86
        x_text = 3.5
    elif hemisph == 'MH':
        ax.set_ylim(-1.3, 0.1)
        ax.set_xlim(11, 20)
        ax.set_yticks(np.arange(-1.2, 0.1, 0.3))
        y_text = -1.25
        x_text = 16.5
    else:
        ax.set_ylim(-0.3, 0.05)
        ax.set_xlim(-0.1, 2)
        ax.set_yticks(np.arange(-0.3, 0.06, 0.1))
        y_text = -0.28
        x_text = 1.5


    ax.text(x_text, y_text, 'r=' + str(np.around(stat_params['corr_coef'], 2)) +
            ' (p=' + str(np.around(stat_params['p_val'], 2)) + ')')

    if idx % ncol == 0:
        ax.legend(loc=6, bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False,
                 labelspacing= 0.8, ncol=1)

    return


def make_plot(data_dict, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    ncols = len(data_dict.keys())

    verb_month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                       9: 'September', 10: 'October', 11: 'November',
                       12: 'December'}

    fig = plt.figure()
    fig.set_size_inches(7.5, 5.5)
    fig.set_dpi(300)

    for n_h, hemisph in enumerate(sorted(data_dict.keys())):
        verb_month = verb_month_dict[data_dict[hemisph].pop('month')]
        obs_dic = data_dict[hemisph].pop('OBS')
        nrows = len(data_dict[hemisph].keys())
        for n_p, proj in enumerate(data_dict[hemisph].keys()):
            make_panel(data_dict[hemisph][proj], nrows, ncols, (n_h + 1) + (n_p + 1) * n_p, obs_dic, verb_month,
                       hemisph, proj)

    fig.suptitle('Mean (x-axis) sea ice area (SIA) and its trend (y-axis)',
                 fontsize='x-large', x=0.38)
    fig.subplots_adjust(left=0.11, right=0.73, top=0.88, bottom=0.1, wspace=0.25, hspace=0.4)

    # for np, proj in enumerate(data_dict[hemisph].keys()):
    #     fig.text(0.37, 0.92-0.47*np, proj, fontsize = 'x-large')

    return


def main(cfg):

    dtsts = Datasets(cfg)

    # here we create the dictionary which afterwards will be plotted
    data_dict = {'NA': {'CMIP6': {}, 'OBS':{}}}

    for hemisph in data_dict.keys():
        # here we choose the month and the part of hemisphere
        month = cfg['month_' + hemisph]
        for project in data_dict[hemisph].keys():
            models = set(dtsts.get_info_list('dataset', project=project))
            for model in models:
                ens_fnames = dtsts.get_info_list('filename', dataset=model, project=project)
                ens_cubelist = iris.load(ens_fnames)
                # calculating sea ice area or extent, depending on seaiceextent
                sea_ice_cubelist = ipcc_sea_ice_diag.calculate_siparam(ens_cubelist, cfg['seaiceextent'])
                #  calculating climatology and regression slope
                ens_stats = calculate_entry_stat(sea_ice_cubelist)
                # calculating average over ensembles in the model
                mod_mean, mod_dec_slope, mod_means, mod_dec_slopes = ens_average(ens_stats)
                # creating a dictionary for a model with mean and decadal slope
                data_dict[hemisph][project][model] = {'mean': mod_mean}
                data_dict[hemisph][project][model]['dec_slope'] = mod_dec_slope
                data_dict[hemisph][project][model]['means_all'] = mod_means
                data_dict[hemisph][project][model]['slopes_all'] = mod_dec_slopes
            #  calculating the statistics for a multi-model ensemble
            if project!='OBS':
                data_dict[hemisph][project] = model_stats(data_dict[hemisph][project])
        # here observations are added, they are already provided as sia. no need to process them the same way as models
        data_dict[hemisph]['month'] = month

    make_plot(data_dict, cfg)

    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_20_scatter', img_ext='.png')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)