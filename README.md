# pySCWP: A python program for wavepacket dynamics using exact and surface hopping methods. 

## Introduction
This repository contains a set of three models describing complex-valued (topological) avoided crossings 
studied in [1].
The models can be solved with either exact wavepacket dynamics using the Fourier transform method [2] or
fewest-switches surface hopping [3] with the gauge fixings implemented in [1] and [4].


## Usage
1. Prepare an input file, use the files in /example/ as a guide.
2. Run the code as
```python
python3 main.py input_file
```
3. The output folder includes the following observables: adiabatic populations, diabatic populations
classical energy (for FSSH simulation only), quantum energy, adiabatic momentum, diabatic momentum, adiabatic population histograms, 
diabatic population histograms. 





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
