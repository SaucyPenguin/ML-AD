# ML-AD
These are the files containing the functions used throughout the project. There are a couple of things that are important to note:

1) The workflow is as follows: adni_file_extract.py --> dbsnp_extract.py --> neural_net.py --> clustering.py --> mapping.py.

2) As you will see, these files contain functions and nothing else. The functions were used at our discretion to pull information and create files containing data types (using the pickle package). As such, there is no clear "pathway" to follow per se; however, the functions provided are satisfactory and work together.

3) The data we used is available through the Alzheimer's Disease Neuroimaging Initiative (ADNI). Due to their use of real patient medical data, we cannot provide our original data files in this repository. However, access to the ADNI database can be applied for and, if approved, used with this code.

4) The package _pybedtools_, as used in the mapping file, is only available through a conda environment. We suggest using this environment throughout the entire project to ensure consistency.

For concerns or inquiries: shrey.sharma.va@gmail.com
