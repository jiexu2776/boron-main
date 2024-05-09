[![DOI](https://zenodo.org/badge/722942640.svg)](https://zenodo.org/doi/10.5281/zenodo.11150471)


# Boron isotope reduction app


<img src="https://user-images.githubusercontent.com/107989499/220860374-dcae816d-a4ad-4fb2-8aeb-84a911ee4170.png" alt="Logo" width="30%" align="right">


Boron isotope reduction app is a web-based streamlit app for boron isotope data reduction. With this app, raw data from Neptune can be reducted to boron isotope values. 

Outlier rejection, background correction, intra-sequence Instrumental drift correction and ablation volume dependent B concentration offset correction are involved.

## Website
Please click here to start:
[Boron isotope reduction app](https://boron-isotopes.streamlit.app/) 

Or please visit: https://boron-isotopes.streamlit.app/


## How to use it
1. Upload all data files from one sequence.

2. Set up parameters for isotopic data: (1). bacground and signal area; (2). outlier factor; (3). the bulge factor for 11B; (4). choose your standard for intra-sequence instrumental drift correction: the name, the ‘A/B/C/D’ inside the name of standard, the regression level.

3. Upload log file from laser.

4. Set up parameters for corrected boron concentration from signal intensity: (1). the regression level; (2).  the depth of selected reference depth and other sample depth; (3). the shape of your spots: circle or squre. (4). split stream or not.

5. Upload your trace element file if you used split stream. (*not necessary)

6. check data from standards and download your final results as a csv file.

<!-- ## Introduction

This program is capable of:

1. Read multiple .exp data files
2. Read additional .csv files
3. Outlier rejection
4. Background correction
5. Intra-sequence Instrumental drift correction
6. Ablation volume dependent B concentration offset correction
7. Combination of calculation results, laser parameters and trace elements results
8. Ready to use final data table -->
