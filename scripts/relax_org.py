# Adapted from https://github.com/materialsvirtuallab/m3gnet#structure-relaxation
import warnings

from pymatgen.core import Lattice, Structure

from m3gnet.models import Relaxer

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)
mo = Structure(Lattice.cubic(3.3), ["Mo", "Mo"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

relaxer = Relaxer()  # This loads the default pre-trained model

relax_results = relaxer.relax(mo, verbose=True)

final_structure = relax_results["final_structure"]
final_energy_per_atom = float(relax_results["trajectory"].energies[-1] / len(mo))

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]:.3f} Å")
print(f"Final energy is {final_energy_per_atom:.3f} eV/atom")
