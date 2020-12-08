<img src="or3d_illustration1.png" width="100%", height="200">

# or3d - 3D Ocean Reconstruction package

This package contains codes and sample notebooks for downloading and processing the SSH mapping and vertical reconstruction tools.
The examples can be run online using: "binder"

## Motivation

The goal is to illustrate in an observation system simulation experiment (OSSE) how to best reconstruct ocean vertical quantities, such as vertical velocity, from partial sea surface height (SSH) observations: conventional altimetry nadirs and future SWOT observations. The methods and codes present in this package are meant to evolve and adapt to different input observations. In anticipation of the SWOT CalVal missions, the ultimate goal of this package is to be prepared to deal with real nadir and SWOT observations when they arrive and to be able to combine these altimetric observations with in situ observations in order to produce the best 3D ocean reconstructions.  
A set of example notebooks is provided (see below) for everyone to be able to start and play as quickly as possible.

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

and should be stored in 'or3D/data/SSH_obs/'.

To start out download the *observation* dataset (dc_obs, 285M) using :

```shell
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz
```
 
and then uncompress the files using tar -xvf <file>.tar.gz. You may also use ftp, rsync or curlto donwload the data.

### Run examples

You can find the examples in 'or3d/data/examples/' or launch them directly from "binder".


## Available methods

### 1. SSH mapping from nadir and SWOT observations

- Optimal Interpolation (see for instance Le Traon and Dibarboure., 1999) 


### 2. Vertical reconstruction from SSH maps 

- Effective surface quasigeostrophic framework (eSQG; see Qiu et al., 2020)



## References

- Qiu, B., Chen, S., Klein, P., Torres, H., Wang, J., Fu, L., & Menemenlis, D. (2020). Reconstructing Upper-Ocean Vertical Velocity Field from Sea Surface Height in the Presence of Unbalanced Motion, Journal of Physical Oceanography, 50(1), 55-79. Retrieved Dec 8, 2020, from https://journals.ametsoc.org/view/journals/phoc/50/1/jpo-d-19-0172.1.xml

- Le Traon, P. Y., & Dibarboure, G. (1999). Mesoscale Mapping Capabilities of Multiple-Satellite Altimeter Missions, Journal of Atmospheric and Oceanic Technology, 16(9), 1208-1223. Retrieved Dec 8, 2020, from https://journals.ametsoc.org/view/journals/atot/16/9/1520-0426_1999_016_1208_mmcoms_2_0_co_2.xml
