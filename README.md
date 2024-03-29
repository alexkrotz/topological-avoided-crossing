# A python program for complex-valued avoided crossings using exact and surface hopping methods. 

## Introduction
This repository contains a set of three models describing complex-valued (topological) avoided crossings 
studied in [1].
The models can be solved with either exact wavepacket (WP) dynamics using the Fourier transform method [2] or
fewest-switches surface hopping (FSSH) [3] with the gauge fixings implemented in [1] and [4].


## Basic Usage
1. Prepare an input file, use the files in /example/ as a guide.
2. Run the code as
```python
python3 main.py input_file
```
3. FSSH calculations can be run on top of each other automatically producing a new 
set of output files with a new "calculation index" appended to the end of each output
filename. 
3. The output folder includes the following observables for FSSH calculations (where n is the calculation index):
   * transmitted adiabatic populations: `pop_adb_n.npy`
   * transmitted diabatic populations: `pop_db_n.npy` 
   * transmitted diabatic momentum: `p_db_n.npy`
   * classical energy: `Ec_n.npy`
   * quantum energy: `Eq_n.npy`
   * upper adiabatic surface population histograms: `adb_1_hist_n.npy`
   * lower adiabatic surface population histograms: `adb_0_hist_n.npy`
   * upper diabatic surface population histograms: `db_1_hist_n.npy`
   * lower diabatic surface population histograms: `db_0_hist_n.npy`
   * grid x-axis: `rxgrid.npy`
   * grid y-axis: `rygrid.npy`
4. The output folder includes the following for WP calculations:
   * time-dependent wavefunction: `psi.npy`
   * list of time steps: `tdat.npy`

## Reproducing Publication Data
To reproduce the data used in [1] refer to the directory /publication_data/ for input
files, directory structure, and auxilliary scripts. 
1. 





## References
[1] Krotz, A. and Tempelaar, R. Treating Geometric Phase Effects in Nonadiabatic Dynamics.
Phys. Rev. A 109, 2024, 032210.

[2] Kosloff, D.; Kosloff, R. A Fourier Method Solution for the Time Dependent Schrödinger Equation
as a Tool in Molecular Dynamics. J. Comput. Phys. 1983, 52 (1), 35–53.

[3] Hammes-Schiffer, S. and Tully, J. C. Proton Transfer in Solution: Molecular Dynamics with
Quantum Transitions. J. Chem. Phys. 101, 1994, 4657–4667.

[4] Miao, G.; Bellonzi, N.; Subotnik, J. An Extension of the Fewest Switches Surface Hopping 
Algorithm to Complex Hamiltonians and Photophysics in Magnetic Fields: Berry Curvature and 
“Magnetic” Forces. J. Chem. Phys. 2019, 150 (12), 124101.
