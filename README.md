# astro-realm : Astronomy - Radio galaxy Evolution Analaysis using ALma and Muse (REALM)

## Background
The module in this repostitory contains methods that I used to carry out a data extraction, analysis and visulation of an astronomical datacubes which is structured as tensors containing pixel counts. 

The telescope used to obtain the data is the incredible ALMA - <a href="https://www.almaobservatory.org/en/home/">Atacama Large Milimeter-submilimeter Array</a> located in the Chajnantor Plateau in the Chilean Andes. ALMA is an interferometer comprising 66 dishes and is a remarkable instrumental that was partly responsible for forming the black-hole image in the Event Horizon Telescope project. 

The array produces datacubes is represented by co-ordinates mapped in the cartesian plane such that (x,y,z) represent (right ascension, declination, velocity). This is displayed in figure below which contains an ALMA datacube with right ascenion, declination and velocity axes. This datacube shows the merging of two stars.

<p align="center">
<img src="alma_datacube.png" height="300">
</p>

In this project, the datacubes imaged a sample of seven radio galaxies. The milimeter/sub-milimeter data allowed us to constrain the kinematics, mass and structure of the molecular gas surrounding the galaxies. 


## Usage
The repository contains Python modules used to carry out a very specific data analysis using ALMA, MUSE and ancillary Hubble Space Telescope (HST), Spitzer Space Telescope (SST) datasets and VLA datasets. 

A test module is included in addition to test datasets from the data sources:

- <a href="http://archive.eso.org/cms.html">ESO Archive</a>
- <a href="https://archive.stsci.edu/hst/">HST Archive</a>
- <a href="https://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzerdataarchives/">SST Archive</a>
- <a href="https://science.nrao.edu/facilities/vla/archive/index">VLA Archive</a>

Python libraries used:
- For astronomy: MPDAF, AstroPy, WCS
- For linear algebra, calculus etc: SciPy, NumPy
- For visualization: Matplotlib
- For model fitting: LMFIT
- Others: Warnings, Itertools, Math


[1] Credit: <a href="http://irfu.cea.fr/Projets/COAST/">Computational Astrophysics at Saclay</a>