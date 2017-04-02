# Network analysis of high-density microelectrode recordings

This repositiory contains the source code as well as the linkt to the example data to replicate all figures. 
The Hana (high density microelectrode array recording analysis) pipeline is open source, see [licence](LICENCE.md).
The example data consists of spike triggered averages and events that were extracted from the raw recordings.  

If you use this library in your research, please cite:

> Bullmann T, et al. Network analysis of high-density microelectrode recordings (????) ????


## How to use

### Source code

Clone 'hdmea' from github and and change to the folder:

```bash
git clone http://github.com/tbullmann/hdmea
cd hdmea
```
### Install requirements

Using conda to create an environment ```hdmea``` and install the requirements:
```bash
conda create --name hdmea python
source activate hdmea 
conda install --file hana/requirements.txt
```

### Folders and data

Now simply type ```bash install.sh``` or make these folder/symlinks as you wish:
```bash
cd publication
mkdir temp 
mkdir figures
```
Download the data: _Under revision_


### Replicate the figures

Assuming you are in ```/publication```, you append your ```PYTHONPATH```, activate the environment ```hdmea``` and run the script from command line:
```bash
export PYTHONPATH=../:$PYTHONPATH     # otherwise cannot see /hana
source activate hdmea 
python all_figures.py 
```
The script should finish after 5~10 minutes. If the temporary files already exist, the figures itself will take only about 2 minutes.
A total of 7 main and 5 supplementary figures can be found as ```*.eps``` and ```*.png``` files in ```/figures```.

In case you are using PyCharm you have to specify the [use of the project interpreter](hdmea_env_in_pycharm.jpg) from the ```hdmea``` environment.

## Folders
Folders structure and important files:
```
.
├── hana
│   ├── requirements.txt
│   ├── LICENCE.md
│   ├── README.md
│   └── ...
├── matlab
│   └── ... (Export function for Matlab)
├── misc           
│   └── ... (Old figures)
├── publication
│   ├── data  
│   ├── temp   
│   ├── figures 
│   ├── all_figures.py
│   └── ...
├── install.sh
├── LICENCE.md
├── README.md
└── ...
```
