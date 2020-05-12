from monty.json import MSONable

import numpy as np
from itertools import combinations

from pymatgen.util.coord import is_coord_subset, coord_list_mapping


SYMMETRY_ERROR = ValueError("Error in calculating symmetry operations. Try using a "
                            "more symmetrically refined input structure. "
                            "SpacegroupAnalyzer(s).get_refined_structure().get_primitive_structure() "
                            "usually results in a safe choice")
class Cluster(MSONable):
    """
    Definition of a cluster. Contains its points and the speices on these points.
    Both of the two!
    """
    def __init__(self,points,nbits,lattice,fractional=True):
        """
        points: 
            2D array-like, n*3, each row is a point's coord
        nbits: 
            A list of indices of the species occupying points.
        lattice: 
            A pymatgen.Lattice object
        fractional:
            If true, points should be in fractional coords;
            Otherwise, should be in cartesian coords. 
        """

        self.lattice = lattice
        self.frac_to_cart = lattice.matrix 
        self.cart_to_frac = np.linalg.inv(lattice.matrix)       
        points = np.array(points)
       
        if not fractional:
            points = points@self.cart_to_frac

        center = np.average(points,axis=0)
        shift = np.floor(center)
         
        self.center = center-shift
        self.points = points-shift
        #These enforces translational symmetry

        self.carts = self.lattice.get_cartesian_coords(self.points)
        self.radius = np.max([np.linalg.norm(r_a-r_b) for r_a,r_b in combinations(carts,2)])

        if len(nbits)!=len(self.points):
            raise ValueError("Not all points occupied!")

        self.nbits = np.array(nbits,dtype=np.int64)
        self.c_id = None

    @property
    def size(self):
        return len(self.points)
 
    def __eq__(self,other):
        if self.lattice != other.lattice:
            return False
        if self.points.shape != other.points.shape:
            return False
        #Not only the same n_bits, but also has to be the same ordering

        other_points = other.points + np.round(self.center - other.center)
        if is_coord_subset(self.points,other_points):
            mapping = coord_list_mapping(self.points,other_points)
            if np.allclose(other.nbits[mapping],self.nbits):
                return True
        return False

    def __str__(self):
        points_str = str(np.round(self.points,2)).replace("\n", " ").ljust(len(self.points) * 21)
        center_str = str(np.round(self.center,2)).replace("\n"," ").ljust(21)
        nbits_str = str(self.nbits).replace("\n"," ").ljust(21)
        return "Cluster: id: {:<3} Radius: {:<4.3} Points: {} Centroid: {} nbits: {}".format(self.c_id, 
                                                                                   self.radius, 
                                                                                   points_str, 
                                                                                   center_str,
                                                                                   nbits_str)

    def __repr__(self):
        return self.__str__()

class ClusterOrbit(MSONable):
    """
    An orbit of clusters that are symmetric under the symmetry operations of the lattice. 
    Each orbit will correspond to a feature and an ECI.
    """
    def __init__(self,base_cluster,structure_symops):
        self.base_cluster = base_cluster
        self.structure_symops = structure_symops
        self.lattice = self.base_cluster.lattice

        self._orbit = None
        self._cluster_symops = None
        self.orb_id = None

    @property
    def cluster_symops(self):
        """
        The symmetry operation of the cluster itself. Contains symops that maps a cluster back
        to itself, represented in an np.int64 array 'inds' so that cluster.points[inds] = 
        new_cluster.points.
        """
        if self._cluster_symops = None:
            orbit = self.orbit
        return self._cluster_symops

    @property
    def orbit(self):
        if self._orbit is None:
            base_operation = np.arange(0,len(self.base_cluster.size),1,dtype=np.int64)
            self._orbit = [self.base_cluster]
            cluster_symops = [base_operation]

            for symop in self.structure_symops:
                new_points = symop.operate_multi(self.base_cluster.points)
                new_cluster = Cluster(new_points,self.base_cluster.nbits,self.lattice)

                if new_cluster == self.base_cluster:
                    cluster_symop = coord_list_mapping(new_cluster.points,self.base_cluster.points)
                    cluster_symop = np.array(cluster_symop,dtype=np.int64)
                    cluster_symops.append(cluster_symop)

                if new_cluster not in self._orbit:
                    self._orbit.append(new_cluster)

        if self._cluster_symops is None:
            self._cluster_symops = cluster_symops

        if len(self._orbit)*len(self._cluster_symops) != len(self.structure_symops):
            raise SYMMETRY_ERROR

        return self._orbit

    @property
    def multiplicity(self):
        return len(self._orbit)
                
    def __eq__(self,other):
        return (self.base_cluster in other.orbit)

    def __str__(self):
        return "ClusterOrbit: id: {:<4} multiplicity: {:<4} symops: {:<4}" \
               " base cluster: {}".format(str(self.orb_id), str(self.multiplicity),\
               str(len(self.cluster_symops)), str(self.base_cluster))

    def __repr__(self):
        return self.__str__()
