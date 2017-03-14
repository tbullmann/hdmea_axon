# Network analysis of high-density microelectrode recordings

Source code as well as example data to replicate the figures. 
The Hana (high density microelectrode array recording analysis) analysis pipeline is open source.
The example data consists of spike triggered averages and events that were extracted from the rawrecordings.  


If you use this library in your research, please cite:

> Bullmann T, et al. Network analysis of high-density microelectrode recordings (????) ????


## How to use

### Source code

Clone 'hdmea' from github:

```bash
git clone http://github.com/tbullmann/hdmea
```

### Folders and data

Change to the folder:
```bash
cd hdmea
```
Now simply type ```bash install.sh``` or make folder or symlinks as you wish:
```bash
cd publication
mkdir temp
mkdir figures
```
Download the data:
```bash
wget "https://www.dropbox.com/s/ahet0hrios57q4a/data.zip"
unzip data.zip
```

### Install requirements

Using conda to create an enviroment and install requirements:
```bash
conda create --name hdmea python
source activate hdmea 
conda install --file hana/requirements.txt
```

### Replicate the figures

Running the script from command line assuming you are in ```/publications```:
```bash
export PYTHONPATH=../:$PYTHONPATH  # otherwise cannot see /hana
source activate hdmea 
pyhton all_figures.py 
```
