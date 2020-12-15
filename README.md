# pydssa 
## Python developements for SWOT satellite analysis


![Illustration](figures/pydssa_illustration1.png)

<>This package contains codes and sample notebooks for downloading and processing the SSH mapping and vertical reconstruction tools.
<>The examples can be run online using: "binder"

The pydssa package is meant to be seen as a toolbox that should be shared and continuously developed by the oceanographic community. 

## Motivation
 

The Surface Water and Ocean Topography (SWOT) satellite mission will launch in 2022 and should provide unprecedented two-dimensional sea surface height observations. In the first three months of data aquisition, SWOT will have a day-repeat fast sampling orbit. This Calibration and Validation (CalVal) phase will offer high temporal resolution data over several orbit crossing points: crossovers. The CalVal phase is a great opportunity for the science community to perform specific studies at fine temporal and spatial scales of oceanographic dynamics to improve our understanding of physical, biological and chemical processes. 

In addition, during the CalVal phase, several in-situ experiments will be held in order to fully take advantage of the SWOT observations. The [Adopt-A-Crossover](https://www.clivar.org/news/swot-‘adopt-crossover’-consortium-has-been-endorsed-clivar) Consortium is responsible for implementing in-situ strategies and sampling of the fine-scale upper ocean processes in the different crossover regions (California Current, Western Pacific, Polar and Sub-polar regions and the Mediterranean). The pydssa package could be a useful toolbox to help the synergy between SWOT and in-situ observations. 
For instance, pydssa has been used to illustrate in an observation system simulation experiment (OSSE) how to reconstruct ocean vertical quantities from SWOT data in the eSQG framework that had vertical stratification parameters calibrated with in-situ observations. 

<> The methods and codes present in this package are meant to evolve and adapt to different input observations. In anticipation of the SWOT CalVal missions, the ultimate goal of this package is to be ready to deal with real nadir and SWOT observations when they arrive and to be able to combine these altimetric observations with in situ observations in order to produce the best 3D ocean reconstructions.  

A set of example notebooks is provided (see below) for everyone to be able to start and play as quickly as possible.

## Installation 

After cloning the pydssa repository, install the package by typing in a terminal (from outside the pydssa repository): 

```shell
pip install -e pydssa/
```

you can then import and use pydssa as any python package

```python
import pydssa
```

## Examples

### Download the example data

#### For examples 1 and 2:

#### For examples 3 and 4:

The observation data are hosted [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/ocean-data-challenges/dc_data1/catalog.html) with the following directory structure

```
.
|-- dc_obs
|   |-- 2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc
|   |-- 2020a_SSH_mapping_NATL60_nadir_swot.nc
|   |-- 2020a_SSH_mapping_NATL60_karin_swot.nc
|   |-- 2020a_SSH_mapping_NATL60_jason1.nc
|   |-- 2020a_SSH_mapping_NATL60_geosat2.nc
|   |-- 2020a_SSH_mapping_NATL60_envisat.nc

``` 

and should be stored in 'pydssa/data/SSH_obs/'.

To start out download the *observation* dataset (dc_obs, 285M) using :

```shell
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz
```
 
and then uncompress the files using tar -xvf <file>.tar.gz. You may also use ftp, rsync or curlto donwload the data.

### Run examples

You can find the examples in 'pydssa/data/examples/' or launch them directly from "binder".


## Available methods

### SSH mapping from nadir and SWOT observations

- Optimal Interpolation (see for instance Le Traon and Dibarboure., 1999) 


### Vertical reconstruction from SSH maps 

- Effective surface quasigeostrophic framework (eSQG; see Qiu et al., 2020)

- Geostrophic kinematic deformation (aka QG Q vector or frontogenesis vector) at the surface only (Hoskins et al., 1978)

## References

- Hoskins, B.J., Draghici, I. and Davies, H.C. (1978), A new look at the ω‐equation. Q.J.R. Meteorol. Soc., 104: 31-38. https://doi.org/10.1002/qj.49710443903

- Le Traon, P. Y., & Dibarboure, G. (1999). Mesoscale Mapping Capabilities of Multiple-Satellite Altimeter Missions, Journal of Atmospheric and Oceanic Technology, 16(9), 1208-1223. Retrieved Dec 8, 2020, from https://journals.ametsoc.org/view/journals/atot/16/9/1520-0426_1999_016_1208_mmcoms_2_0_co_2.xml

- Qiu, B., Chen, S., Klein, P., Torres, H., Wang, J., Fu, L., & Menemenlis, D. (2020). Reconstructing Upper-Ocean Vertical Velocity Field from Sea Surface Height in the Presence of Unbalanced Motion, Journal of Physical Oceanography, 50(1), 55-79. Retrieved Dec 8, 2020, from https://journals.ametsoc.org/view/journals/phoc/50/1/jpo-d-19-0172.1.xml

