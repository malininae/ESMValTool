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

def create_map_plot(data_dic, mixns, ano_map_cb, cfg):

    b_min = np.floor(np.min([np.percentile(ano_map_cb.data, 1, method='closest_observation'), mixns['min']])) 
    b_max = np.ceil(np.max([np.percentile(ano_map_cb.data, 99, method='closest_observation'), mixns['max']]))
    if cfg['cbar_positive']: 
        levels = np.arange(b_min, b_max+0.1, 0.1)
        cmap = 'Reds'
    else:
        abs_max = np.abs([b_min, b_max]).max()
        levels = np.arange(-1*abs_max, abs_max+0.1, 0.1)
        cmap = 'RdBu_r'

    for dataset in data_dic.keys(): 
        fig_map, ax_map = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        ax_map = ax_map.flatten()

        fig_map.set_size_inches(7., 9.)
        fig_map.set_dpi(200)

        model_data = list()
        weights = list()

        for i in range(0,len(data_dic[dataset])):
            model_data.append(data_dic[dataset][i].data)
            weights.append(data_dic[dataset][i].attributes['ensemble_weight']*
                            data_dic[dataset][i].attributes['reverse_dtsts_n'])
        weights = np.array(weights)
        model_data = np.array(model_data)
        if dataset == 'Multi-Model-Mean':
            mean_arr = np.average(model_data, axis=0, weights=weights)
        else:
            mean_arr = np.average(model_data, axis=0)
        
        map_c = ax_map[0].contourf(data_dic[dataset][i].coord('longitude').points,
                                   data_dic[dataset][i].coord('latitude').points,
                                   mean_arr, levels=levels, extend='both', cmap=cmap)
        ax_map[0].set_title(dataset+' '+cfg['title_var_label'])

        iplt.contourf(ano_map_cb, axes=ax_map[1], levels=levels, extend='both', cmap=cmap)
        ax_map[1].set_title('ERA5 '+ cfg['title_var_label'])

        [a.coastlines(linewidth=0.5) for a in ax_map]
        fig_map.subplots_adjust(left=0.02, bottom=0.1, right=0.99, top=0.92, wspace=0.02, hspace=0.1)
        cax = fig_map.add_axes([0.1,0.07,0.8,0.015])
        fig_map.colorbar(map_c, cax=cax, orientation='horizontal', label=cfg['cbar_label'] +', '+ cfg['cbar_units'])

        fig_map.savefig(os.path.join(cfg['plot_dir'], 'map_anomalies_'+dataset+'.'+cfg['output_file_type']))

        if cfg['ratio']:
            fig_rat, ax_rat = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

            fig_rat.set_size_inches(7., 5.)
            fig_rat.set_dpi(200)

            map_rat = iplt.contourf(1/(ano_map_cb/mean_arr), axes=ax_rat, levels=np.arange(0.2, 3.1, 0.2), extend='both', cmap='Oranges')

            ax_rat.set_title('Ratio of '+cfg['title_var_label']+' ('+dataset+'/ERA5)')
            ax_rat.coastlines(linewidth=0.5)

            fig_rat.subplots_adjust(left=0.02, bottom=0.15, right=0.99, top=0.92, wspace=0.02, hspace=0.1)
            cax = fig_rat.add_axes([0.1,0.13,0.8,0.015])
            fig_rat.colorbar(map_rat, cax=cax, orientation='horizontal', label=cfg['cbar_label'] +' ratio')

            fig_rat.savefig(os.path.join(cfg['plot_dir'], 'ratio_anomalies_'+dataset+'.'+cfg['output_file_type']))


    return


def main(cfg):

    input_data = cfg['input_data']
    
    groups = group_metadata(input_data.values(), 'variable_group', sort=True)
    obs_info = groups.pop('obs')
    ano_map_cb = iris.load_cube(obs_info[0]['filename'])

    groups_l = list(groups.keys())

    plotting_dic = {}
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
                mins.append(mod_cb.data.min()); maxs.append(mod_cb.data.max())
            plotting_dic[group][dataset] =  mod_cubelist
        mixns[group]['max'] = np.asarray(maxs).max()
        mixns[group]['min'] = np.asarray(mins).min()
        plotting_dic[group]['Multi-Model-Mean'] =  ens_cubelist

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    create_map_plot(plotting_dic['txx_all'], mixns['txx_all'], ano_map_cb, cfg)

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
