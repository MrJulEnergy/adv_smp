import zntrack
import ase
import matplotlib.pyplot as plt
from pathlib import Path


class CompareDFT(zntrack.Node):
    """Analyse convergence of different DFT calculations.

    Parameters
    ----------
    data : list[list[ase.Atoms]]
        List of lists of ASE Atoms objects. Each list corresponds to a different DFT calculation
        from the same structures.
    mgrids : list[int]
        List of integers representing the grid sizes used in the DFT calculations.
        The length of this list should match the number of different DFT calculations
        in `data`.

    Attributes
    ----------
    plots : Path
        Path to the directory where the plots will be saved.
    """

    data: list[list[ase.Atoms]] = zntrack.deps()
    mgrids: list[int] = zntrack.params()

    plots: Path = zntrack.outs_path(zntrack.nwd / "plots")

    def run(self):
        if len(self.data) != len(self.mgrids):
            raise ValueError("Length of data and mgrids must match.")

        for frames in self.data:
            for atoms in frames:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                # TODO

        self.plots.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        # TODO

        fig.savefig(self.plots / "energy.png")