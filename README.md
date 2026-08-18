# Py2DFTIR_GPU
The data analysis code of the newly developed wide-band 2D FT-IR. The original 2D FT-IR was fabricated by Nissin Kikai Co..
A 2D FT-IR capable of wider bandwidth imaging was developed, based on the design of the original 2D FT-IR　(Zhao et al., SPIE proc., https://doi.org/10.1117/12.3019640). We achieved faster processing through parallel computing in this development.

This code requires a Python environment with the necessary scientific computing libraries installed. GPU acceleration is supported and requires an appropriate GPU computing environment, including compatible GPU drivers and CUDA-related libraries.

First, the users run the code of 2D WFT-IR_Create_Tmap.py, and the file "T_map_lookup.npz" should be created.
Then, they specify the sample and reference data names with directory in the code of 2D WFT-IR_gpu_ver3.py, and run it.
