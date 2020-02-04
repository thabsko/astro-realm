# radio_galaxy_phd_project-2

## Background
The module in this repostitory contains methods that I used to carry out a data extraction, analysis and visulation of an astronomical datacubes. The datacubes are structured as tensors containing pixel counts from the detector. 

The telescope used to obtain the data is the incredible ALMA - <a href="https://www.almaobservatory.org/en/home/">Atacama Large-milimeter-submilimeter Array</a> located in the Chajnantor Plateau in the Chilean Andes. ALMA is an interferometer comprising 66 dishes and is a remarkable instrumental that was partly responsible for forming the black-hole image in the Event Horizon Telescope project. The telescope produces datacubes that have dimensions mapped such that (x,y,z) = (right ascension, declination, velocity) as shown in Fig. 1 [1]. 

In this project, the datacubes imaged a sample of seven radio galaxies. The milimeter/sub-milimeter data allowed us to constrain the kinematics, mass and structure of the molecular gas surrounding the galaxies. 

<figure>
<img align="middle" src="alma_datacube.png" height="300">
  <figcaption>Fig. 1: An ALMA datacube showing the right ascenion, declination and velocity axes. This datacube shows the merging of two stars.</figcaption>
</figure>

## Usage
The repository contains Python modules used to carry out the data analysis. Although the functions are customised to my own user case, it is possible to follow the method and use my implementations in your own code. 

Python libraries used:
- For astronomy: MPDAF, AstroPy
- For linear algebra, calculus etc: SciPy, NumPy
- For visualization: Matplotlib
- For model fitting: LMFIT
- Others: Warnings, Itertools, Math


[1] Credit: <a href="http://irfu.cea.fr/Projets/COAST/">Computational Astrophysics at Saclay</a>