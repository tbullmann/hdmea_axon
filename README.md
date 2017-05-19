# Network analysis of high-density microelectrode recordings

![Example neuron 5](neuron5.gif) ![Example graph](synaptic_delay_graph.png)

This repositiory contains the source code as well as the links to the example data to replicate all [figures](publication/README.md). 
The Hana (high density microelectrode array recording analysis) pipeline is open source, see [licence](LICENCE.md).
The example data (approx. 4 GB) consists of spike triggered averages and events that were extracted from the raw recordings.  

If you use this library in your research, please cite our paper ([BioRXiv](http://biorxiv.org/content/early/2017/05/18/139436)):

> Bullmann T, Radivojevic M, Huber S, Deligkaris K, Hierlemann A, Frey U (2017) Network analysis of high-density microelectrode recordings. _Submitted_

```bib
@article {Bullmann139436,
	author = {Bullmann, Torsten and Radivojevic, Milos and Huber, Stefan T. and Deligkaris, Kosmas and Hierlemann, Andreas and Frey, Urs},
	title = {Network Analysis Of High-Density Microelectrode Recordings},
	year = {2017},
	doi = {10.1101/139436},
	URL = {http://biorxiv.org/content/early/2017/05/18/139436},
	eprint = {http://biorxiv.org/content/early/2017/05/18/139436.full.pdf},
	journal = {bioRxiv}
}
```


## How to use

### Source code

Clone ```hdmea``` from GitHub and and change to the folder:

```bash
git clone http://github.com/tbullmann/hdmea
cd hdmea
```
### Install requirements

Using conda to create an environment ```hdmea``` (or any other name) and install the requirements:
```bash
conda create --name hdmea python
source activate hdmea 
conda install --file hana/requirements.txt
```

### Folders and data

Create folder by typing ```bash install.sh```, or make these folder/symlinks as you wish:
```bash
cd publication
mkdir temp 
mkdir figures
mkdir data
```
Download the [data from google drive](https://drive.google.com/open?id=0B-u65ZxPB5iQNFBoa192WmpIQW8) and copy into the data folder. For the proper folder structure see section 'Folders' below.

### Replicate the figures

Assuming you are in ```/publication```, you append your ```PYTHONPATH```, activate the environment ```hdmea``` and run the script from command line:
```bash
export PYTHONPATH=../:$PYTHONPATH     # otherwise cannot see /hana
source activate hdmea 
python all_figures.py 
```
The script takes about one hour for the first run on the full data set (7 networks, 4GByte). 
If the temporary files already exist, the figures itself will take only about 3 minutes.
A total of 7 main and 3 supplementary figures can be found as ```*.eps``` and ```*.png``` files in ```/figures```.

### Replicate the movies

You need to install a renderer for gif ([ImageMagick](https://www.imagemagick.org/script/download.php)) or mpeg ([ffmpeg](https://ffmpeg.org/download.html)). 

Continue by typing:
```bash
python all_animations.py
```
This script takes about 15 minutes. A total of 23 movies can be found as ```*.gif``` and/or ```*.mp4``` files in ```temp/culture1```.

![Example neuron 5](neuron5.gif)


## Using PyCharm

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
│   │   ├── culture1  
│   │   └── ...
│   ├── temp   
│   │   ├── culture1  
│   │   └── ...
│   ├── figures 
│   ├── all_figures.py
│   ├── all_animations.py
│   └── ...
├── install.sh
├── LICENCE.md
├── README.md
└── ...
```
