"""Utilities to parse composition constraints from input options."""

from smol.cofe.space.domain import get_species
from smol.moca.utils.occu import get_dim_ids_by_sublattice


def parse_species_constraints(d, bits, sl_sizes):
    """Parse the constraint to species concentrations.

    Args:
        d(dict|list/tuple of dict):
            Dictionary of restrictions. Each key is a representation of a
            species, and each value can either be a tuple of lower and
            upper-limits, or a single float of upper-limit of the species
            atomic fraction. Number must be between 0 and 1, which means
            If a list of dict provided, each dict in the list constrains
            on a sub-lattice composition. This is sometimes necessary when
            you wish to constrain vacancy concentrations on some specific
            sub-lattices. d must be given in the same ordering or "bits",
            if d is given by each sub-lattice.
            If only one dict is provided, the bounds number will be atomic
            fraction of the particular species in all the sub-lattices that
            allows it! (number_of_species/sum(sl_size_of_allowed_sublatt)).
        bits(list[list[Species|Vacancy|Element]]):
            Species on each sublattice. Must be exactly the same as used in
            CompSpace initializer.
        sl_sizes(list[int]):
            size of sub-lattices in a primitive cell. Must be given in the
            same ordering as bits.
    Return:
        list, list: constraints in CompSpace readable format.
    """

    def recursive_parse(inp):
        p = {}  # Saved keys must not be objects.
        if isinstance(inp, (list, tuple)):
            return [recursive_parse(o) for o in inp]
        else:
            for key, val in inp.items():
                if isinstance(val, (list, tuple)):
                    if len(val) != 2:
                        raise ValueError(
                            "Species concentration constraints provided "
                            "as tuple, but length of tuple is not 2."
                        )
                    if val[1] < val[0]:
                        raise ValueError(
                            "Species concentration constraints provided "
                            "as tuple, but lower bound > upper bound."
                        )
                    if val[1] < 0 or val[1] > 1 or val[0] < 0 or val[0] > 1:
                        raise ValueError(
                            "Provided species concentration limit must " "be in [0, 1]!"
                        )
                    p[get_species(key)] = tuple(val)
                else:
                    if val < 0 or val > 1:
                        raise ValueError(
                            "Provided species concentration limit must " "be in [0, 1]!"
                        )
                    p[get_species(key)] = (0, val)
        return p

    parsed = recursive_parse(d)
    dim_ids = get_dim_ids_by_sublattice(bits)
    n_dims = sum([len(sub_bits) for sub_bits in bits])
    constraints_leq = []
    constraints_geq = []
    if isinstance(parsed, list):
        for sub_parsed, sub_bits, sub_dim_ids, sl_size in zip(
            parsed, bits, dim_ids, sl_sizes
        ):
            for sp in sub_parsed:
                dim_id = sub_dim_ids[sub_bits.index(sp)]
                con = [0 for _ in range(n_dims)]
                con[dim_id] = 1
                constraints_geq.append((con, sub_parsed[sp][0] * sl_size))  # per-prim.
                constraints_leq.append((con, sub_parsed[sp][1] * sl_size))
    else:
        for sp in parsed:
            con = [0 for _ in range(n_dims)]
            r_leq = 0
            r_geq = 0
            for sub_bits, sub_dim_ids, sl_size in zip(bits, dim_ids, sl_sizes):
                if sp in sub_bits:
                    dim_id = sub_dim_ids[sub_bits.index(sp)]
                    con[dim_id] = 1
                    r_geq += parsed[sp][0] * sl_size
                    r_leq += parsed[sp][1] * sl_size
            constraints_geq.append((con, r_geq))
            constraints_leq.append((con, r_leq))

    return constraints_leq, constraints_geq


def parse_generic_constraint(d_left, right, bits):
    """Parse more generic constraint.

    Parse one constraint at a time.
    Args:
        d_left(dict| list(dict)):
            Dictionary that records the left-hand side of the
            constraint equation. Each key is a species or species
            string, while each value is the pre-factor of the
            corresponding species number in the constraint equation.
            If given in list of dictionary, each dictionary in the
            list will constrain a corresponding sub-lattice.
            If given in a single dictionary, that means the amount
            of species that appear in this dictionary will be
            constrained regardless of sub-lattice.
            Note: numbers must be given as per primitive cell.
        right(float):
            Right-hand side of the equation. Must be given as per
            primitive cell. When parsing an equality constraint, must
            be integer.
        For example: 1 n_Li + 2 n_Ag = 1 can be specified as:
            d_left = {"Li": 1, "Ag": 2}
            right = 1
        bits(list[Species|Element|Vacancy]):
            Species on each sublattice. Must be exactly the same as used
            in CompSpace initializer.
    Note:
        Currently when parsing an equality constraint, only integers are allowed
        on both left and right side.
    Returns:
        tuple(list, int):
           The parsed constraint in CompSpace format.
    """

    def recursive_parse(inp):
        p = {}  # Saved keys must not be objects.
        if isinstance(inp, (list, tuple)):
            return [recursive_parse(o) for o in inp]
        else:
            for key, val in inp.items():
                p[get_species(key)] = val

        return p

    parsed = recursive_parse(d_left)
    dim_ids = get_dim_ids_by_sublattice(bits)
    n_dims = sum([len(sub_bits) for sub_bits in bits])
    con = [0 for _ in range(n_dims)]
    if isinstance(parsed, list):
        for sub_parsed, sub_bits, sub_dim_ids in zip(parsed, bits, dim_ids):
            for sp in sub_parsed:
                dim_id = sub_dim_ids[sub_bits.index(sp)]
                con[dim_id] = sub_parsed[sp]
    else:
        for sp in parsed:
            for sub_bits, sub_dim_ids in zip(bits, dim_ids):
                if sp in sub_bits:
                    dim_id = sub_dim_ids[sub_bits.index(sp)]
                    con[dim_id] = parsed[sp]

    return con, right
