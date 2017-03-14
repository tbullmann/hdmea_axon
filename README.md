# Network analysis of high-density microelectrode recordings

Source code as well as example data to replicate the figures. 
The Hana (high density microelectrode array recording analysis) analysis pipeline is open source.
The example data consists of spike triggered averages and events that were extracted from the rawrecordings.  


If you use this library in your research, please cite:

> Bullmann T, et al. (????) ????


## Installation

### Installing hana 

Clone hdmea source folder 'hdmea'.

```bash
git clone http://github.com/tbullmann/hdmea
cd hdmea
git submodule init
git submodule update --recursive
```

### Installing requirements

(...)

## Replicating the figures

```bash
cd publication
```
Make folders (or symlink) for figures and temporary file
```bash
mkdir temp
mkdir figures
```
Download the data from dropbox
```bash
wget "https://www.dropbox.com/s/.../data.zip"
unzip data.zip
```
Running the script
```bash
pyhton all_figures.py 
```
