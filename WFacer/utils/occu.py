"""Generate random occupancy."""
import numpy as np


def get_random_occupancy_from_counts(ensemble, counts):
    """Generate random occupancy from species counts.

    Args:
        ensemble(Ensemble):
            An ensemble object to generate occupancy in.
        counts(1D arrayLike):
            Species composition in "counts" format.
            See smol.moca.composition.

    Returns:
        np.ndarray:
            An encoded occupancy array.
    """
    n_species = 0
    occu = np.zeros(ensemble.num_sites, dtype=int) - 1
    for sublatt in ensemble.sublattices:
        n_sublatt = counts[n_species : n_species + len(sublatt.encoding)]
        if np.sum(n_sublatt) != len(sublatt.sites):
            raise ValueError(
                f"Composition: {counts} does not match "
                f"super-cell size on sub-lattice: {sublatt}!"
            )
        occu_sublatt = [
            code for code, n in zip(sublatt.encoding, n_sublatt) for _ in range(n)
        ]
        np.random.shuffle(occu_sublatt)
        occu[sublatt.sites] = occu_sublatt
        n_species += len(sublatt.encoding)
    if np.any(occu < 0):
        raise ValueError(
            f"Given composition: {counts}\n "
            f"or sub-lattices: {ensemble.sublattices}\n "
            f"cannot give a valid occupancy!"
        )
    return occu
