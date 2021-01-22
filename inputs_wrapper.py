#TODO

        #Calculation of bits, sublat_list, is_charges_ce
        #previous_ce are no longer computed here.
        #I have moved it to ElementalWrapper.

        bits = get_allowed_species(self.prim)
        if sublat_list is not None:
            #User define sublattices, which you may not need very often.
            self.sublat_list = sublat_list
            self.sl_sizes = [len(sl) for sl in self.sublat_list]
            self.bits = [bits[sl[0]] for sl in self.sublat_list]
        else:
            #Automatic sublattices, same rule as smol.moca.Sublattice:
            #sites with the same compositioon are considered same sublattice.
            self.sublat_list = []
            self.bits = []
            for s_id,s_bits in enumerate(bits):
                if s_bits in self.bits:
                    s_bits_id = self.bits.index(s_bits)
                    self.sublat_list[s_bits_id].append(s_id)
                else:
                    self.sublat_list.append([s_id])
                    self.bits.append(s_bits)
            self.sl_sizes = [len(sl) for sl in self.sublat_list]



            if radius is not None and len(radius)>0:
                self.radius = radius
            else:
                d_nns = []
                for i,site1 in enumerate(self.prim):
                    d_ij = []
                    for j,site2 in enumerate(self.prim):
                        if j<i: continue;
                        if j>i:
                            d_ij.append(site1.distance(site2))
                        if j==i:
                            d_ij.append(min([self.prim.lattice.a,self.prim.lattice.b,self.prim.lattice.c]))
                    d_nns.append(min(d_ij))
                d_nn = min(d_nns)
    
                self.radius= {}
                # Default cluster radius
                self.radius[2]=d_nn*4.0
                self.radius[3]=d_nn*2.0
                self.radius[4]=d_nn*2.0


            c_spc = ClusterSubspace.from_cutoffs(self.prim,self.radius,\
                                    basis = self.basis_type)

            if self.is_charged_ce:
                c_spc.add_external_term(EwaldTerm())
                coef = np.zeros(c_spc.num_corr_functions+1)
                coef[-1] = 1.0
            else:
                coef = np.zeros(c_spc.num_corr_functions)
