import numpy as np
import pandas as pd
import math
import copy


class TBModel(object):
    def __init__(self, directory, lattice_vecs,atom_specs, atom_redcoords,orb_redcoords=None,orbs_orth = False,min_hopping_dist=None,recip=False):
        self.directory = directory
        self._a = np.array(lattice_vecs)
        self.numAtoms = len(atom_specs)
        self.elements = atom_specs
        self.primAtoms = np.array(atom_redcoords)
        self.orbs_orth = orbs_orth
        self.recip = recip
        if min_hopping_dist == None:
            self.set_min = False
        else:
            self.set_min = True
            self.min_hopping_dist = min_hopping_dist
        #print(self._a)
        self.cartAtoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), self.primAtoms)
        #print(self.cartAtoms)
        self.orbatomnum = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3])
        self.orbatomnum = np.array([0,0,0,0,1,1,1,1])

        if orb_redcoords == None:
            check_file = directory+"wannier90_centres.xyz"
            filedata = open(check_file)
            filelines = filedata.readlines()
            num_orbs = int(filelines[0])
            orb_cartcoords = []
            for linenum in range(2,2+num_orbs):
                line=filelines[linenum]
                orb_pos = [float(i) for i in line.strip().split()[1:]]
                orb_cartcoords.append(orb_pos)
            orb_cartcoords = np.array(orb_cartcoords)
            orb_redcoords = _cart_to_red((self._a[0], self._a[1], self._a[2]),orb_cartcoords)
        self.orb_redcoords = orb_redcoords
        #print(self.orb_redcoords)

    def read_TBparams(self,file = "test_hr.dat.txt"):
        #get TB params from a file formatted like wannier90_hr.dat
        filename = self.directory + file
        #print(filename)
        filedata = open(filename)
        filelines = filedata.readlines()
        num_orbs = int(filelines[1])
        self.num_orbs = num_orbs
        # print("number of orbitals:",num_orbs)
        if self.orbs_orth == True:
            wk_pnts = int(filelines[2])
            first_line = 3 + math.ceil(wk_pnts/15) #100 #3+num_orbs #12
        else:
            first_line = 3+num_orbs
            self.aeorb_overlap = np.zeros((num_orbs, num_orbs),dtype=np.complex_)
            line_num = 1
            for orb1 in range(num_orbs):
                line_num += 1
                line = filelines[line_num]
                self.aeorb_overlap[orb1] = [float(i) for i in line.strip().split()]
            # print(self.aeorb_overlap)
        last_line = len(filelines)
        ## print(filelines[first_line])
        count = 0
        num_each_dir = 3# abs(int(filelines[first_line].split()[0]))
        self.num_each_dir = num_each_dir
        num_trans = num_each_dir*2+1
        self.num_trans = num_trans
        ## print(num_trans)
        #generate list of the displacement between translations
        vec_to_trans = np.zeros((num_trans,num_trans,num_trans,3))
        for x in range(num_trans):
            for y in range(num_trans):
                for z in range(num_trans):
                    vec_to_trans[x,y,z] = [x-num_each_dir,y-num_each_dir,z-num_each_dir]
        self.vec_to_trans = vec_to_trans
        #read in the TB parameters
        TB_params = np.zeros((num_orbs,num_orbs,num_trans,num_trans,num_trans), dtype=np.complex_)
        for line in filelines[first_line:]:
            count += 1
            info = line.split()
            if abs(int(info[0])) <=num_each_dir and abs(int(info[1])) <=num_each_dir and abs(int(info[2])) <=num_each_dir:
                trans1 = int(info[0])+num_each_dir
                trans2 = int(info[1])+num_each_dir
                trans3 = int(info[2])+num_each_dir
                orb1 = int(info[3])-1
                orb2 = int(info[4])-1
                value = float(info[5]) +float(info[6])*1.0j
                if TB_params[orb1,orb2,trans1,trans2,trans3] != 0:
                    pass
                    # print("already set TB param")
                #only set if orbitals are not on the same atom
                same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                ## print(same_atom)

                TB_params[orb1, orb2, trans1, trans2, trans3] = value
                #if (same_atom < 0.001).all() and (trans1==0 and trans2==0 and trans3==0) and orb1!=orb2:
                    ## print("set to zero!", same_atom)
                    #TB_params[orb1,orb2,trans1,trans2,trans3] = 0
            #else:
            #    # print("didn't include all hoppings",info)
        self.TB_params = TB_params
        ## print(info)
        ## print(float(info[5]))
        ## print(TB_params[:,:,0,0,0])
        ## print(count)

    def set_hoppings(self,value,orb1,orb2,trans,set_conjugate=True):
        (trans1,trans2,trans3) = trans
        (trans1, trans2, trans3) = (trans1+self.num_each_dir,trans2+self.num_each_dir,trans3+self.num_each_dir)
        self.TB_params[orb1,orb2,trans1,trans2,trans3] = value
        if set_conjugate == True:
            (ntrans1,ntrans2,ntrans3) = trans
            (ntrans1, ntrans2, ntrans3) = (-ntrans1+self.num_each_dir,-ntrans2+self.num_each_dir,-ntrans3+self.num_each_dir)
            self.TB_params[orb2, orb1, ntrans1, ntrans2, ntrans3] = value

    def get_hoppings(self,info=None):
        if info == None:
            return self.TB_params
        else:
            pass
            # print("will work out later")

    def read_orbitals(self):
        first_file = self.directory+"orbital0.txt"
        file = open(first_file)
        filelines = file.readlines()
        #self._grid = self.grid#[24,24,30]
        self._grid = np.array([int(i) for i in filelines[1].strip().split()])
        self.grid = self._grid
        self._a1 = np.array([float(i) for i in filelines[2].strip().split()])
        self._a2 = np.array([float(i) for i in filelines[2 + 1].strip().split()])
        self._a3 = np.array([float(i) for i in filelines[2 + 2].strip().split()])
        self._a = np.array([self._a1,self._a2,self._a3])
        ## print(self._grid)
        ## print(self._a)
        self.orbitals = {}

        for orb in range(self.num_orbs):
            currentFile = self.directory + 'orbital'+str(orb)+".txt"
            # print(currentFile)

            # get wavefunction data
            df = pd.read_csv(currentFile, header=None, delimiter=r'\s+', dtype=float, skiprows=6,
                             nrows=int(math.ceil(self._grid[0] * self._grid[1] * self._grid[2] / 6)))  # , skipfooter=2)
            data = df.values.flatten()

            data = data[~np.isnan(data)]
            # # print(len(data))
            wavefunc = data.reshape((self._grid[0], self._grid[1], self._grid[2]))
            self.orbitals[str(orb)] = wavefunc


        #get minimum radius to nearest atom for each atom
        #self.grid = [24,24,30]#[24,24,30]
        X, Y, Z = np.mgrid[0:(1 - 1 / self.grid[0]):self.grid[0] * 1j, 0:(1 - 1 / self.grid[1]):self.grid[1] * 1j,0:(1 - 1 / self.grid[2]):self.grid[2] * 1j]
        self.cartXYZ = _red_to_cart((self._a[0], self._a[1], self._a[2]),np.array([X.flatten(), Y.flatten(), Z.flatten()]).transpose())
        self.cartXYZ = self.cartXYZ.transpose()
        ## print(self.cartXYZ[0])
        ## print(self.cartXYZ[1])
        ## print(self.cartXYZ[2])
        self.min_xyzreduced = {}
        for atomnum in range(self.numAtoms):
            atom = self.cartAtoms[atomnum]
            all_atoms = []

            # get atom positions translated to surrounding unit cells
            for trans1 in range(3):
                for trans2 in range(3):
                    for trans3 in range(3):
                        vec = (trans1 - 1) * self._a[0] + (trans2 - 1) * self._a[1] + (trans3 - 1) * self._a[2]
                        all_atoms.append(vec + atom)
            ## print(all_atoms)
            point = self.cartXYZ
            min_rad = np.ones((len(self.cartXYZ[0]))) * 100
            min_phi = np.ones((len(self.cartXYZ[0]))) * 100
            min_theta = np.ones((len(self.cartXYZ[0]))) * 100

            min_x = np.ones((len(self.cartXYZ[0]))) * 100
            min_y = np.ones((len(self.cartXYZ[0]))) * 100
            min_z = np.ones((len(self.cartXYZ[0]))) * 100
            vectors_to_atoms = np.ones((len(self.cartXYZ[0]), 3, 27)) * 100
            numatm = 0
            # get vectors to surrounding atoms in spherical coordinates
            for atm in all_atoms:
                vec = point - np.array([atm]).transpose()
                rad = np.linalg.norm(vec, axis=0)
                rad[rad==0] = 0.00001
                # # print(rad.shape)
                theta = np.arccos(vec[2] / rad)
                phi = np.arctan2(vec[1], vec[0])

                bool1 = [rad < min_rad]
                min_rad[bool1] = rad[bool1]
                min_theta[bool1] = theta[bool1]
                min_phi[bool1] = phi[bool1]
                min_x[bool1] = vec[0][bool1]
                min_y[bool1] = vec[1][bool1]
                min_z[bool1] = vec[2][bool1]
                vectors_to_atoms[:, :, numatm] = np.array([rad, phi, theta]).transpose()
                numatm += 1
            #self.min_rad[str(atomnum)] = min_rad
            #self.min_radthetaphi[str(atomnum)] = [min_rad,min_theta,min_phi]
            #self.min_xyz[str(atomnum)] = [min_x,min_y,min_z]
            self.min_xyzreduced[str(atomnum)] = _cart_to_red((self._a[0], self._a[1], self._a[2]), np.array([min_x,min_y,min_z]).transpose()).transpose()
        ## print(self.min_xyzreduced[str(0)][:25])
        ## print(self.min_xyzreduced[str(1)][:25])

        if self.recip == True:
            # get reciprocal orbitals
            # include all gpoints
            kpt = [0,0,0]
            self.gpoints = self.generate_gpnts(kpt) # np.array([gX.flatten(), gY.flatten(), gZ.flatten()]).transpose()
            self.num_gpnts = len(self.gpoints)
            # print("gpts:", self.num_gpnts, self.gpoints)

            keys = list(self.orbitals.keys())
            recip_orbs = {}
            for orb in keys:
                recip_orbs[orb] = np.zeros((self.num_gpnts), dtype=np.complex_)
                mesh = np.fft.fftn(self.orbitals[orb])
                mesh = np.fft.fftshift(mesh)
                for gind, gp in enumerate(self.gpoints):
                    t = tuple(gp.astype(np.int_) + (self.grid / 2).astype(np.int_))
                    recip_orbs[orb][gind] = mesh[t]
            self.recip_orbs = recip_orbs#self.real_to_recip(self.orbitals)


            # get reciprocal orbitals centered at (0,0,0)
            grided_recip_orbs = {}
            for orb in range(self.num_orbs):
                atomnum = self.orbatomnum[orb]
                center = self.primAtoms[atomnum]
                exp_term = np.exp(2j * np.pi * np.dot(self.gpoints, center))
                coeffs = self.recip_orbs[str(orb)] * exp_term  # center orbital so that the G->G+k doesn't effect e^iGR
                mesh = np.zeros(tuple(self.grid), dtype=np.complex_)
                for gp, coeff in zip(self.gpoints, coeffs):
                    t = tuple(gp.astype(np.int_) + (self.grid/ 2).astype(np.int_))
                    mesh[t] = coeff
                grided_recip_orbs[orb] = mesh
            self.centered_recip_orbs = grided_recip_orbs

    def real_to_recip(self,real_orbs):
        keys = list(real_orbs.keys())
        recip_orbs = {}
        kpt = 0
        for orb in keys:
            recip_orbs[orb] = np.zeros((self.grid), dtype=np.complex_)
            mesh = np.fft.fftn(real_orbs[orb])
            mesh = np.fft.fftshift(mesh)
            recip_orbs[orb] = mesh
            #for gind, gp in enumerate(self.gpoints[kpt]):
            #    t = tuple(gp.astype(np.int_) + (self.grid / 2).astype(np.int_))
            #    recip_orbs[orb][gind] = mesh[t]

        return recip_orbs

    def generate_gpnts(self,kpt):  # from pymatgen.io.vasp.outputs.Wavecar
        self._C = 0.262465831
        self.encut = 5200
        self.vol = np.dot(self._a[0, :], np.cross(self._a[1, :], self._a[2, :]))

        # calculate reciprocal lattice
        b = np.array(
            [
                np.cross(self._a[1, :], self._a[2, :]),
                np.cross(self._a[2, :], self._a[0, :]),
                np.cross(self._a[0, :], self._a[1, :]),
            ]
        )
        b = 2 * np.pi * b / self.vol
        self.b = b
        self._nbmax = [0,0,0]
        self._nbmax[0] = np.int(self.grid[0] / 3)
        self._nbmax[1] = np.int(self.grid[1] / 3)
        self._nbmax[2] = np.int(self.grid[2] / 3)

        gamma=False
        if gamma:
            kmax = self._nbmax[0] + 1
        else:
            kmax = 2 * self._nbmax[0] + 1

        gpoints = []
        extra_gpoints = []
        extra_coeff_inds = []
        G_ind = 0
        for i in range(2 * self._nbmax[2] + 1):
            i3 = i - 2 * self._nbmax[2] - 1 if i > self._nbmax[2] else i
            for j in range(2 * self._nbmax[1] + 1):
                j2 = j - 2 * self._nbmax[1] - 1 if j > self._nbmax[1] else j
                for k in range(kmax):
                    k1 = k - 2 * self._nbmax[0] - 1 if k > self._nbmax[0] else k
                    if gamma and ((k1 == 0 and j2 < 0) or (k1 == 0 and j2 == 0 and i3 < 0)):
                        continue
                    G = np.array([k1, j2, i3])
                    v = kpt + G
                    g = np.linalg.norm(np.dot(v, self.b))
                    E = g ** 2 / self._C
                    if E < self.encut:
                        gpoints.append(G)
                        if gamma and (k1, j2, i3) != (0, 0, 0):
                            extra_gpoints.append(-G)
                            extra_coeff_inds.append(G_ind)
                        G_ind += 1
        return gpoints

    def get_ham(self,kpt,scaledTBparams = False):
        ham = np.zeros((self.num_orbs,self.num_orbs),dtype=np.complex_)
        vec_to_orbs = np.zeros(self.vec_to_trans.shape)
        ## print("check same:", self.orbitals[str(0)][12,12,:])

        #self.Aij[kpt] = kdep_Aij
        '''
        # construct k-dependant Sij and Aij, only different if psuedo basis orbital are not orthogonal
        kdep_Sij = np.zeros((self.num_orbs, self.num_orbs))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                atmnum1 = self.orbatomnum[orb1]
                atmnum2 = self.orbatomnum[orb2]
                if atmnum1 == atmnum2:
                    # extra exponential factor cancels out so same regardless of kpoint
                    kdep_Sij[orb1, orb2] = aeorb_overlap[orb1, orb2]
                else:
                    # need to recalculate overlap of distorted orbitals
                    new_orbs_multplied = np.conj(expon_term[orb1] * orth_orbs[str(orb1)]) * expon_term[orb2] * \
                                         orth_orbs[str(orb2)]
                    distorb_overlap = periodic_integral_3d(new_orbs_multplied, self._a, self.gridxyz)
                    kdep_Sij[orb1, orb2] = distorb_overlap

                # good approximation for large unit cells but not small with multiple vectors to nearest atom of one type
                # exp_fac = np.exp((2.0j) * np.pi * np.dot(kpoints[kpt], (self.orbpos[orb1] - self.orbpos[orb2]).transpose()))
                # kdep_Sij[orb1,orb2] = aeorb_overlap[orb1,orb2]*exp_fac
        eigenvalj, kdep_Dij = np.linalg.eig(kdep_Sij)
        # check correctness of eigen
        # construct Aij
        kdep_Aij = np.zeros((self.num_orbs, self.num_orbs))
        for j in range(self.num_orbs):
            kdep_Aij[:, j] = kdep_Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
        self.Aij[kpt] = kdep_Aij
        '''
        if scaledTBparams == True:
            scaledTB = np.zeros((self.num_orbs,self.num_orbs,self.num_trans,self.num_trans,self.num_trans), dtype=np.complex_)
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs[:,:,:] = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans**3,3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                exp_fac = np.exp(-2j*np.pi*np.dot(kpt,hold_vecs))
                if self.set_min == True:
                    exp_fac[dist_to_orbs > self.min_hopping_dist] = 0
                exp_fac = np.reshape(exp_fac,(self.num_trans,self.num_trans,self.num_trans))
                ham[orb1,orb2] = np.sum(self.TB_params[orb1,orb2]*exp_fac)
                if scaledTBparams == True:
                    scaledTB[orb1,orb2] = self.TB_params[orb1,orb2]*exp_fac

        if scaledTBparams == True:
            return scaledTB
        ## print(ham)

        if self.orbs_orth == True:
            (eigval, eigvec) = np.linalg.eigh(ham)
        '''
        else:
            if self.recip==False:
                expon_term = np.zeros((self.numAtoms, self.grid[0], self.grid[1], self.grid[2]), dtype=np.complex_)
                for atm in range(self.numAtoms):
                    min_reducedcoord_diff = np.array(self.min_xyzreduced[str(atm)])
                    # print(kpt,min_reducedcoord_diff[:,0])
                    # print(np.dot(kpt, min_reducedcoord_diff[:,0]))
                    term = np.exp(2j * np.pi * np.dot(kpt, min_reducedcoord_diff))
                    expon_term[atm] = np.reshape(term, self.grid)
            else:
                gpoints = self.generate_gpnts(kpt)
                num_gpnts = len(gpoints)
                print("gpts:", num_gpnts)#, gpoints)
                g_coords = []
                for gp in gpoints:
                    t = tuple(gp.astype(np.int_) + (self.grid / 2).astype(np.int_))
                    g_coords.append(t)
                g_coords = np.array(g_coords)
                #g_coords = self.g_coords
                gplusk_coords = g_coords.transpose() + np.array([kpt]).transpose()
                adjusted_orbs = {}
                for orb in range(self.num_orbs):
                    atomnum = self.orbatomnum[orb]
                    center = self.primAtoms[atomnum]
                    exp_term = np.exp(-2j * np.pi * np.dot(gpoints, center))  # not e^-iGR because is complex conjugate
                    test_coeffs = ndimage.map_coordinates(self.centered_recip_orbs[orb].real, gplusk_coords, mode="wrap")
                    test_coeffs = test_coeffs + 1j * ndimage.map_coordinates(self.centered_recip_orbs[orb].imag, gplusk_coords, mode="wrap")
                    # print("is same?", coeffs, test_coeffs)
                    adjusted_orbs[str(orb)] = test_coeffs * exp_term

            kdep_Sij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex_)
            for orb1 in range(self.num_orbs):
                for orb2 in range(self.num_orbs):
                    atmnum1 = self.orbatomnum[orb1]
                    atmnum2 = self.orbatomnum[orb2]
                    if atmnum1 == atmnum2:
                        # extra exponential factor cancels out so same regardless of kpoint
                        kdep_Sij[orb1, orb2] = self.aeorb_overlap[orb1, orb2]
                    else:
                        # need to recalculate overlap of distorted orbitals
                        if self.recip== False:
                            new_orbs_multplied = np.conj(expon_term[atmnum1]*self.orbitals[str(orb1)])*expon_term[atmnum2]*self.orbitals[str(orb2)]
                            #print("here",new_orbs_multplied[12,:,12])
                            distorb_overlap = periodic_integral_3d(new_orbs_multplied, self._a, self.grid)
                            #print(distorb_overlap)
                        else:
                            new_orbs_multplied = np.conj(adjusted_orbs[str(orb1)]) * adjusted_orbs[str(orb2)]
                            distorb_overlap = reciprocal_integral(new_orbs_multplied, self._a, self.grid)
                        kdep_Sij[orb1, orb2] = distorb_overlap
            
            eigenvalj, kdep_Dij = np.linalg.eig(kdep_Sij)
            # check correctness of eigen
            # construct Aij
            kdep_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex_)
            for j in range(self.num_orbs):
                kdep_Aij[:, j] = kdep_Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
            #get ham for orthogonal eigenvectors
            conj_Aij = np.conj(kdep_Aij).transpose()
            AtHA = np.matmul(conj_Aij,np.matmul(ham,kdep_Aij))
            (eigval,eigvec) = np.linalg.eigh(AtHA)
        '''

        #print(eigval)
        #print(eigvec[:,19])
        return eigval,eigvec

    '''
    def plot_sphOrbitals(self):
        figures = {}
        axises = {}

        from skimage.measure import marching_cubes

        #self.one_orbitalWF = self.lowdin_orth(self.one_orbitalWF)
        grid = self.orbitals[str(0)].shape
        for orb in range(len(self.orbitals)):
            orbital = self.orbitals[str(orb)]#**2*np.exp(-6*np.reshape(self.min_rad[str(self.orbatomnum[orb])],self.gridxyz))
            figures[str(orb)] = plt.figure()
            axises[str(orb)] = figures[str(orb)].add_subplot(111, projection='3d')

            iso_val = np.max(orbital)/2
            verts, faces, _, _ = marching_cubes(orbital, iso_val, spacing=(1/grid[0],1/grid[1],1/grid[2]))
            new_verts = np.array([verts[:, 0], verts[:, 1], verts[:, 2]]).transpose()
            cart_verts = _red_to_cart((self._a[0], self._a[1], self._a[2]), new_verts)
            axises[str(orb)].plot_trisurf(cart_verts[:, 0], cart_verts[:, 1], faces, cart_verts[:, 2], color='g', lw=1)

            if (orbital < -iso_val).any():
                iso_val_neg = -iso_val
                vertsn, facesn, _, _ = marching_cubes(orbital, iso_val_neg, spacing=(1/grid[0],1/grid[1],1/grid[2])) #
                new_vertsn = np.array([vertsn[:, 0], vertsn[:, 1], vertsn[:, 2]]).transpose()
                cart_vertsn = _red_to_cart((self._a[0], self._a[1], self._a[2]), new_vertsn)
                axises[str(orb)].plot_trisurf(cart_vertsn[:, 0], cart_vertsn[:, 1], facesn, cart_vertsn[:, 2], color='pink', lw=1)

            axises[str(orb)].view_init(elev=90, azim=0)
            axises[str(orb)].set_xlim()

            plt.show()
    '''
    def get_bandstructure(self,return_evals = False,kpoints=None):
        if kpoints == None:
            # get kpath
            import pymatgen.core.structure as struc
            import pymatgen.symmetry.kpath as kpath
            pymat_struc = struc.Structure(self._a, self.elements, self.primAtoms)
            # print(pymat_struc)
            kpts_obj = kpath.KPathSeek(pymat_struc)
            kpoints = kpts_obj.get_kpoints(8, coords_are_cartesian=False)
            self.num_kpts = len(kpoints[0])
            self.kpoint_coords = np.array(kpoints[0])
            self.kpoint_labels = np.array(kpoints[1])
            # high_sym_pnts = kpath_obj.kpath
            # print(self.kpoint_coords)
            # print(kpoints[1])


        eigvals = np.zeros((self.num_kpts,self.num_orbs))
        eigvecs = np.zeros((self.num_kpts,self.num_orbs,self.num_orbs))


        for kpt in range(self.num_kpts):
            # print(kpt,self.kpoint_coords[kpt])
            eigvals[kpt],eigvecs[kpt] = self.get_ham(self.kpoint_coords[kpt])
        self.evals = eigvals
        self.eigvecs = eigvecs
        if return_evals == True:
            return eigvals

    def plot_hopping(self):
        # fig = plt.figure()
        import matplotlib.pyplot as plt
        actual_hoppings = copy.deepcopy(self.TB_params)
        for orb in range(self.num_orbs):
            center = int((self.num_trans - 1) / 2)
            actual_hoppings[orb][orb][center][center][center] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs, self.num_orbs, self.num_trans, self.num_trans, self.num_trans))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:, :, :] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs, (self.num_trans ** 3, 3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs, axis=1)
                dist_to_orbs = np.reshape(dist_to_orbs, (self.num_trans, self.num_trans, self.num_trans))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        # print("hopping dist:", hopping_distance)
        # print("num hops:", num_hops)
        hopping_strength[hopping_strength == 0] = 10 ** (-6)

        plt.scatter(hopping_distance, np.log10(np.abs(hopping_strength)))
        plt.xlim((0, 30))
        plt.show()
        plt.close()
'''
    def plotBS(self, ax=None, selectedDot=None, plotnew=None):
        """
        :param ax (matplotlib.pyplot axis): axis to save the bandstructure to, otherwise generate new axis
        :param selectedDot (1D integer array): gives kpoint and band index of the dot selected to make green circle
                                eg: [3,4]
        :return: matplotlib.pyplot axis with bandstructure plotted
        """
        fermi_energy = 0#5.619735  # best si: 7.089 #used to set the valence band maximum to zero
        #print("tick info:", self.k_node, self.k_label)

        #reformat data to find nodes and stop repeating them
        num_nodes = 1
        nodes = [0]
        labels = [self.kpoint_labels[0]]
        kpt_line = np.arange(self.num_kpts)
        next_label = ''
        for kpt_ind,name in enumerate(self.kpoint_labels[1:-1]):
            if name != '' and name != next_label:
                nodes.append(kpt_ind-num_nodes+2)
                next_label = self.kpoint_labels[kpt_ind+2]
                if name == next_label:
                    labels.append(name)
                else:
                    labels.append(name+"|"+next_label)
                kpt_line[kpt_ind+2:] = kpt_line[kpt_ind+2:]-1
                num_nodes += 1


        print(num_nodes)
        print(labels)


        #initialize figure and axis
        fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(111)
        else:
            ax = ax

        #plot the bands, i loops over each band
        for i in range(self.num_orbs):
            ax.plot(kpt_line, self.evals[:,i] - fermi_energy, c='gray', picker=True, label=str(i), linewidth=3)
            #if plotnew == True:
            #    ax.plot(self.k_dist, self.new_evals[i] - fermi_energy, c='red', picker=True, label=str(i), linewidth=3)
            #k_dist and evals[i] are 1D arrays of equal length


        #plot the vertical lines for high symmetry points
        for n in range(num_nodes):
            ax.axvline(x=nodes[n], linewidth=0.5, color='k')

        #make the selected dot green
        #if selectedDot != None:
        #    ax.plot(self.k_dist[selectedDot[0]], self.evals[selectedDot[1]][selectedDot[0]][0] - fermi_energy, 'o',
        #            c='limegreen', picker=True, label=str(i), markersize=10)

        #make plot look nice
        ax.set_xlabel("Path in k-space")  # ,fontsize=20)
        ax.set_ylabel("Energy (eV)")  # ,fontsize=20)
        #ax.set_xlim(self.k_dist[0], self.k_dist[-1])
        ax.set_xticks(nodes)
        ax.set_xticklabels(labels)  # ,fontsize=20)
        #ax.set_ylim(-50, 30)
        #fig.tight_layout()
        #plt.close()
        plt.show()
        return ax

    def plot_hopping(self):
        #fig = plt.figure()
        actual_hoppings = copy.deepcopy(self.TB_params)
        for orb in range(self.num_orbs):
            center = int((self.num_trans-1)/2)
            actual_hoppings[orb][orb][center][center][center] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs,self.num_orbs,self.num_trans,self.num_trans,self.num_trans))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans**3,3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                dist_to_orbs = np.reshape(dist_to_orbs,(self.num_trans,self.num_trans,self.num_trans))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        print("hopping dist:",hopping_distance)
        print("num hops:",num_hops)
        hopping_strength[hopping_strength == 0] = 10**(-6)

        plt.scatter(hopping_distance,np.log10(np.abs(hopping_strength)))
        plt.xlim((0,30))
        plt.show()

    def get_DFT_bandstruc(self,directory):

        from pymatgen.io.vasp.outputs import Eigenval
        bandstrucKPT = Eigenval(directory+'EIGENVAL')
        band_kpts = bandstrucKPT.kpoints
        num_kpts = bandstrucKPT.nkpt
        # get DFT eigenvalues
        DFTeigvals = np.array(list(bandstrucKPT.eigenvalues.values()))[0,:,:,0]
        occupations = np.array(list(bandstrucKPT.eigenvalues.values()))[0,:,:,1]
        print("kpts:",band_kpts)
        print("evals:",DFTeigvals.shape,DFTeigvals)
        # get TB model eigenvalues
        self.num_kpts = num_kpts
        self.kpoint_coords = band_kpts
        TBeigvals = np.array(self.get_bandstructure(return_evals=True))
        # error data #in the future use occupations
        num_bands = 8
        num_VB = 4
        diffEig = np.abs(DFTeigvals[:,:num_bands] - TBeigvals)
        sqDiff = np.square(diffEig)
        wan_dis = (np.sum(sqDiff[:,:num_VB])/num_kpts/4)**(1/2)
        avgVBerr = np.average(diffEig[:,:num_VB])
        avgMCBerr = np.average(diffEig[:,num_VB])
        avgCBerr = np.average(diffEig[:,num_VB:])
        print("average error in Valence Bands:", avgVBerr)
        print("average error in Bottom CB:", avgMCBerr)
        print("average error in Conduc Bands:", avgCBerr)
        print("band distance in Valence Bands:", wan_dis)
        #num_bands = len(DFTeigvals[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for band in range(num_bands):
            #for kpt in range(self.num_kpts):
            #    ax.plot(kpt, eigvals[kpt][band], 'o', c='blue', markersize=projections[kpt, band] * 6 + 0.5)
            ax.plot(range(num_kpts), TBeigvals[:, band] , c='red')
            ax.plot(range(num_kpts), DFTeigvals[:, band] , c='black')
        plt.show()
'''

def periodic_integral_3d(f,prim_vec,n,multiple_wfs=False):
    func = np.array(f).transpose() #3D float array
    grid = n #[24,24,33]
    #max_val = np.max(func)
    nz = grid[2]
    ny = grid[1]
    nx = grid[0]
    cross_ab = np.cross(prim_vec[0],prim_vec[1])
    supercell_volume = abs(np.dot(cross_ab,prim_vec[2]))
    indivcell_volume = supercell_volume/nz/ny/nx
    volume = indivcell_volume
    integral = (func[0,:,:]+func[nz-1,:,:])*0.5
    tosum = (func[:-1,:,:]+func[1:,:,:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(nz-1):
    #    integral += (func[:,:, k] + func[:,:, k + 1]) * 0.5
    afterz_integral = integral
    integral = (afterz_integral[0,:] + afterz_integral[ny - 1,:]) * 0.5
    tosum = (afterz_integral[:-1,:]+afterz_integral[1:,:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(ny - 1):
    #    integral += (afterz_integral[:, k] + afterz_integral[:, k + 1]) * 0.5
    aftery_integral = integral

    integral = (aftery_integral[0] + aftery_integral[nx - 1]) * 0.5
    tosum = (aftery_integral[:-1]+aftery_integral[1:])*0.5
    integral = integral + np.sum(tosum,axis=0)
    #for k in range(nx - 1):
    #   integral += (aftery_integral[k] + aftery_integral[k + 1]) * 0.5
    total_integral = integral * volume

    return total_integral.transpose()

def reciprocal_integral(f,prim_vec,n):
    integral = np.sum(f)
    #scale so that is same as real space integral
    grid = n
    nz = grid[2]
    ny = grid[1]
    nx = grid[0]
    cross_ab = np.cross(prim_vec[0], prim_vec[1])
    supercell_volume = abs(np.dot(cross_ab, prim_vec[2]))
    indivcell_volume = supercell_volume / nz / ny / nx
    volume = indivcell_volume
    return integral *volume /nz / ny / nx

def _cart_to_red(tmp,cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    #  ex: prim_coord = _cart_to_red((a1,a2,a3),cart_coord)
    (a1,a2,a3)=tmp
    # matrix with lattice vectors
    cnv=np.array([a1,a2,a3])
    # transpose a matrix
    cnv=cnv.T
    # invert a matrix
    cnv=np.linalg.inv(cnv)
    # reduced coordinates
    red=np.zeros_like(cart,dtype=float)
    for i in range(0,len(cart)):
        red[i]=np.dot(cnv,cart[i])
    return red

def _red_to_cart(prim_vec,prim_coord):
    """
    :param prim_vec: three float tuples representing the primitive vectors
    :param prim_coord: list of float tuples for primitive coordinates
    :return: list of float tuples for cartesian coordinates
            ex: cart_coord = _red_to_cart((a1,a2,a3),prim_coord)
    """
    (a1, a2, a3) = prim_vec
    prim = np.array(prim_coord)
    # cartesian coordinates
    cart = np.zeros_like(prim_coord, dtype=float)
    # for i in range(0,len(cart)):
    cart = [a1] * np.array([prim[:, 0]]).T + [a2] * np.array([prim[:, 1]]).T + [a3] * np.array([prim[:, 2]]).T
    return cart


#test new model!
#PbO
#latticev = [[4.058,0,0],[0,4.058,0],[0,0,5.343]]
#atoms = ["Pb","Pb","O",'O']
#atom_coords = [[0.5,0,0.223],[0,0.5,0.777],[0.5,0.5,0],[0,0,0]]
#orb_coords = [[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],[0.5,0,0.223],
#              [0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],[0,0.5,0.777],
#              [0.5,0.5,0],[0.5,0.5,0],[0.5,0.5,0],[0.5,0.5,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

#Si
#latticev = [[0,2.7344,2.7344],[2.7344,0,2.7344],[2.7344,2.7344,0]]
#atoms = ["Si",'Si']
#atom_coords = [[0.25,0.25,0.25],[0,0,0]]
#orb_coords = [[0.25,0.25,0.25],[0.25,0.25,0.25],[0.25,0.25,0.25],[0.25,0.25,0.25],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

#GaAs
#latticev = [[0,2.7344,2.7344],[2.7344,0,2.7344],[2.7344,2.7344,0]]
#atoms = ["Ga",'As']
#atom_coords = [[0.,0.,0.],[0.75,0.75,0.75]]
#orb_coords = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0.75,0.75,0.75],[0.75,0.75,0.75],[0.75,0.75,0.75],[0.75,0.75,0.75]]

# my TB Silicon model
#direct = "final_Si/uniform/7x7x7/"#"final_Si/uniform/7x7x7/" #'final_PbO/uniform/noTimeSym/'"PbO_fmtedUNKtest/"
#test = TBModel(direct,latticev,atoms,atom_coords,orb_coords,orbs_orth=False,recip=False,min_hopping_dist=25.)
#test.read_TBparams()
#test.read_orbitals()
#test.test_if_kpt_are_same()
#BandstrucDir = "final_Si/bandstruc/"#"final_PbO/bandstruc/"#"final_Si/bandstruc/"
#test.get_DFT_bandstruc(BandstrucDir)

# wannier90 model
#direct = "PbO_fmtedUNKtest/" #"Si/" #'final_PbO/uniform/noTimeSym/'"PbO_fmtedUNKtest/"
#test = TBModel(direct,latticev,atoms,atom_coords,orbs_orth=True,min_hopping_dist=25.)
#file = "wannier90_hrorig.dat"
#test.read_TBparams(file)
#BandstrucDir = "final_PbO/bandstruc/"
#test.get_DFT_bandstruc(BandstrucDir)


#test.plot_sphOrbitals()
#(eig, vec) = test.get_ham([0,0,0])
#print(eig,vec)
#test.plot_hopping()
#test.get_bandstructure()
#test.plotBS()