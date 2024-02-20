# Branch specific documentation

## Information 

This branch was created by [@malininae](https://github.com/malininae) to run CCCma rapid extreme event attribution system. For now only heatwaves are implemented. This branch differs from the official [ESMValTool main branch](https://github.com/ESMValGroup/ESMValTool/tree/main), but the recipes plan to be moved there in the foreseable future. 

The information on scientific content of the heatwave recipes can be found in [this paper](https://doi.org/10.1016/j.wace.2024.100642). 

## Guidelines how to use

The **environment.yml** differes from the official one. Please be aware that additional **climextremes** package has been downloaded from pip. This is a **python** wrapper around **R** package. Thus, to insure the correct instalation after installing the conda environment using 
`mamba env create --name rapid_event_climex --file environment.yml`
and activating that environment, one should run **R** byt typing in the command line `R`. 

After that, in the **R promt** one should run `library('climextRemes')`. In case the package was not installed properly, than please install it by typing in **R prompt** `install.packages('climextRemes')` and follow the instructions from the prompt. Please, don't forget to save the image after quiting R. 

Otherwise for further use, please consult official ESMValTool **ESMValTool_README.md**. 
