import numpy as np
import pymatgen.core.structure as struc
import pymatgen.symmetry.kpath as kpath
from pymatgen.io.vasp.outputs import Eigenval
#import matplotlib
#matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import pandas as pd
#from skimage.measure import marching_cubes
from scipy import interpolate, integrate #,ndimage
import copy
import plotly.graph_objects as go

class TBModel(object):
    def __init__(self, directory, orbs_orth = False,min_hopping_dist=None,min_overlap_dist=None,num_each_dir = np.array([3,3,3])):
        self.directory = "app/cohp/" + directory
        self.read_input()
        self.orbs_orth = orbs_orth
        if min_hopping_dist == None:
            self.set_min = False
        else:
            self.set_min = True
            self.min_hopping_dist = min_hopping_dist
            if min_overlap_dist == None:
                self.min_overlap_dist = min_hopping_dist
            else:
                self.min_overlap_dist = min_overlap_dist
        #print(self._a)
        self.cartAtoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), self.primAtoms)
        #print(self.cartAtoms)
        # generate reciprocal lattice
        self.vol = np.dot(self._a[0, :], np.cross(self._a[1, :], self._a[2, :]))
        b = np.array(
               [np.cross(self._a[1, :], self._a[2, :]),
                np.cross(self._a[2, :], self._a[0, :]),
                np.cross(self._a[0, :], self._a[1, :])])
        b = 2 * np.pi * b / self.vol
        self._b = b
        #print("recip vecs:",b)

        cross_ab = np.cross(self._a[0], self._a[1])
        supercell_volume = abs(np.dot(cross_ab, self._a[2]))
        self.cell_vol = supercell_volume

        #read in params
        self.num_each_dir = num_each_dir
        self.read_TBparams(num_each_dir=num_each_dir)
        if orbs_orth == False:
            self.read_overlaps(num_each_dir=num_each_dir)
        #self.read_recip_rad_orbs()

    def read_input(self,file = "tb_input.txt"):
        # print("hello")
        # import os
        # print(os.listdir())
        filename = self.directory + file
        #print(filename)
        filedata = open(filename)
        filelines = filedata.readlines()
        num_lines = len(filelines)
        for lind,line in enumerate(filelines):
            if "lattice_vecs" in line:
                lattice_start = lind+1
            if "atoms" in line:
                atom_start = lind+1
            if "orbitals" in line:
                orb_start = lind+1
            if "fermi" in line:
                efermi = float(line.strip().split()[2])
            if "shift" in line:
                energy_shift = float(line.strip().split()[2])
        self.efermi = efermi
        self.energy_shift = energy_shift
        #print("fermi energy:",self.efermi)
        #print("energy shift:",self.energy_shift)
        a1 = [float(i) for i in filelines[lattice_start].strip().split()]
        a2 = [float(i) for i in filelines[lattice_start+1].strip().split()]
        a3 = [float(i) for i in filelines[lattice_start+2].strip().split()]
        self._a = np.array([a1,a2,a3])
        #print("prim vecs:",self._a)

        atoms_pos = []
        elems = []
        for lind in range(atom_start,orb_start-1):
            vals = filelines[lind].strip().split()
            if len(vals) == 4:
                atom_pos = [float(i) for i in vals[:3]]
                elem = vals[3]
                atoms_pos.append(atom_pos)
                elems.append(elem)
        #print(atoms_pos,elems)
        self.elements = np.array(elems)
        self.primAtoms = np.array(atoms_pos)
        self.numAtoms = len(self.elements)

        orbs_pos = []
        orbatomnum = []
        orbatomname = []
        exactorbtype = []
        for lind in range(orb_start,num_lines):
            vals = filelines[lind].strip().split()
            if len(vals) == 5:
                orb_pos = [float(i) for i in vals[:3]]
                num = int(vals[3])
                exact_orb = vals[4]
                orbs_pos.append(orb_pos)
                orbatomnum.append(num)
                orbatomname.append(self.elements[num])
                exactorbtype.append(exact_orb)
        #print(orbs_pos,orbatomnum)
        self.orbatomnum = np.array(orbatomnum)
        self.orbatomname = orbatomname
        self.orb_redcoords = np.array(orbs_pos)
        self.exactorbtype = np.array(exactorbtype)

    def read_TBparams(self,file = "TBparams.txt",num_each_dir = np.array([2,2,2])):
        #get TB params from a file formatted like wannier90_hr.dat
        if self.orbs_orth == True:
            file = "orth_tbparams.txt"
        filename = self.directory + file
        #print(filename)
        filedata = open(filename)
        filelines = filedata.readlines()
        num_orbs = int(filelines[1])
        self.num_orbs = num_orbs
        #print(num_orbs)
        if self.orbs_orth == True:
            first_line = 100 #3+num_orbs #12
        else:
            first_line = 3+num_orbs
            self.aeorb_overlap = np.zeros((num_orbs, num_orbs),dtype=np.complex_)
            line_num = 1
            for orb1 in range(num_orbs):
                line_num += 1
                line = filelines[line_num]
                self.aeorb_overlap[orb1] = [float(i) for i in line.strip().split()]
            #print(self.aeorb_overlap)
        last_line = len(filelines)
        #print(filelines[first_line])
        count = 0
        #num_each_dir = np.array([2,2,3])# abs(int(filelines[first_line].split()[0]))
        num_trans = num_each_dir*2+1
        self.num_trans = num_trans
        #print(num_trans)
        #generate list of the displacement between translations
        vec_to_trans = np.zeros((num_trans[0],num_trans[1],num_trans[2],3))
        for x in range(num_trans[0]):
            for y in range(num_trans[1]):
                for z in range(num_trans[2]):
                    vec_to_trans[x,y,z] = [x-num_each_dir[0],y-num_each_dir[1],z-num_each_dir[2]]
        self.vec_to_trans = vec_to_trans
        atomic_energies = []
        #read in the TB parameters
        TB_params = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex_)
        for line in filelines[first_line:]:
            count += 1
            info = line.split()
            if abs(int(info[0])) <=num_each_dir[0] and abs(int(info[1])) <=num_each_dir[1] and abs(int(info[2])) <=num_each_dir[2]:
                trans1 = int(info[0])+num_each_dir[0]
                trans2 = int(info[1])+num_each_dir[1]
                trans3 = int(info[2])+num_each_dir[2]
                orb1 = int(info[3])-1
                orb2 = int(info[4])-1
                value = float(info[5]) +float(info[6])*1.0j
                if TB_params[orb1,orb2,trans1,trans2,trans3] != 0:
                    #print("already set TB param")
                    pass
                #only set if orbitals are not on the same atom
                same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                #print(same_atom)

                TB_params[orb1, orb2, trans1, trans2, trans3] = value
                if (same_atom < 0.001).all() and (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1!=orb2 and abs(value) > 0.0001:
                    # print("Same atom orbital hopping term that should be zero!", orb1, orb2, value)
                    #TB_params[orb1,orb2,trans1,trans2,trans3] = 0
                    pass
                if (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1==orb2:
                    #print("onsite term:",value)
                    atomic_energies.append(value)
        #print("the atomic orbital energies!",atomic_energies)
        self.TB_params = TB_params

    def read_overlaps(self,file = "overlaps.txt",num_each_dir = np.array([2,2,2])):
        #get TB params from a file formatted like wannier90_hr.dat
        filename = self.directory + file
        #print(filename)
        filedata = open(filename)
        filelines = filedata.readlines()
        num_orbs = int(filelines[1])
        self.num_orbs = num_orbs
        #print(num_orbs)
        if self.orbs_orth == True:
            first_line = 100 #3+num_orbs #12
        else:
            first_line = 3+num_orbs
            self.aeorb_overlap = np.zeros((num_orbs, num_orbs),dtype=np.complex_)
            line_num = 1
            for orb1 in range(num_orbs):
                line_num += 1
                line = filelines[line_num]
                self.aeorb_overlap[orb1] = [float(i) for i in line.strip().split()]
            #print(self.aeorb_overlap)
        last_line = len(filelines)
        #print(filelines[first_line])
        count = 0
        #num_each_dir = np.array([2,2,3])# abs(int(filelines[first_line].split()[0]))
        num_trans = num_each_dir*2+1
        self.num_trans = num_trans
        #print(num_trans)
        #generate list of the displacement between translations
        atomic_energies = []
        #read in the TB parameters
        overlaps_params = np.zeros((num_orbs,num_orbs,num_trans[0],num_trans[1],num_trans[2]), dtype=np.complex_)
        for line in filelines[first_line:]:
            count += 1
            info = line.split()
            if abs(int(info[0])) <=num_each_dir[0] and abs(int(info[1])) <=num_each_dir[1] and abs(int(info[2])) <=num_each_dir[2]:
                trans1 = int(info[0])+num_each_dir[0]
                trans2 = int(info[1])+num_each_dir[1]
                trans3 = int(info[2])+num_each_dir[2]
                orb1 = int(info[3])-1
                orb2 = int(info[4])-1
                value = float(info[5]) + float(info[6])*1.0j
                if overlaps_params[orb1,orb2,trans1,trans2,trans3] != 0:
                    #print("already set TB param")
                    pass
                #only set if orbitals are not on the same atom
                same_atom = np.abs(np.array(self.orb_redcoords[orb1]) - np.array(self.orb_redcoords[orb2]))
                #print(same_atom)

                overlaps_params[orb1, orb2, trans1, trans2, trans3] = value
                if (same_atom < 0.001).all() and (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1!=orb2 and abs(value) > 0.0001:
                    #print("Same atom orbital overlap that should be zero!", orb1, orb2, value)
                    pass
                    #TB_params[orb1,orb2,trans1,trans2,trans3] = 0
                if (int(info[0])==0 and int(info[1])==0 and int(info[2])==0) and orb1==orb2:
                    #print("onsite term:",value)
                    atomic_energies.append(value)
        #print("the atomic orbital self overlaps!",atomic_energies)
        self.overlaps_params = overlaps_params

    def set_hoppings(self,value,orb1,orb2,trans):
        (trans1,trans2,trans3) = trans
        self.TB_params[orb1,orb2,trans1,trans2,trans3] = value

    def get_hoppings(self,info=None):
        if info == None:
            return self.TB_params
        else:
            pass
            #print("will work out later")

    def read_recip_rad_orbs(self):
        filename = self.directory+"recip_rad_orbs.txt"
        file = open(filename)
        filelines = file.readlines()
        first_line = filelines[0].strip().split()
        num_orbs = int(first_line[0])
        grid_size = int(first_line[1])
        max_rad = float(first_line[2])
        rad_grid = np.linspace(0,max_rad,num=grid_size,endpoint=False)

        #find orbitals
        orb_start = []
        sph_harm_key = []
        centers = []
        for lind,line in enumerate(filelines):
            if len(line.strip().split()) > 1:
                if line.strip().split()[0] == "new_orb":
                    #print(lind)
                    orb_start.append(lind)
                    harm_key = int(line.strip().split()[1])
                    sph_harm_key.append(harm_key)
                    center = [float(i) for i in filelines[lind+1].strip().split()[1:]]
                    centers.append(center)
        #print(centers)
        harm_to_spd = np.array(["s","p","p","p","d","d","d","d","d"])
        sph_harm_key = np.array(sph_harm_key)
        self.sph_harm_key = sph_harm_key
        self.orbtype = harm_to_spd[sph_harm_key]
        #print("orb types:",self.orbtype)
        self.prim_orb_centers = centers

        #print("num orbs", num_orbs, len(orb_start))
        orb_start.append(len(filelines))
        recip_rad_orb = {}
        for orb in range(num_orbs):
            start = orb_start[orb]+2
            end = orb_start[orb+1]
            num_rows = end-start
            # get wavefunction data
            df = pd.read_csv(filename, header=None, delimiter=r'\s+', dtype=float, skiprows=start, nrows=num_rows)  # , skipfooter=2)
            data = df.values.flatten()
            data = data[~np.isnan(data)]
            #print(data)
            recip_rad_orb[orb] = np.array(data)
        self.recip_rad_orbs = recip_rad_orb
        self.recip_rad_grid = rad_grid

    def generate_gpnts(self,kpt):  # from pymatgen.io.vasp.outputs.Wavecar
        self._C = 0.262465831
        self.encut = 520
        self.vol = self.cell_vol
        # calculate reciprocal lattice

        self.b = self._b
        self._nbmax = [0,0,0]
        self._nbmax[0] = int(np.around(np.linalg.norm(self._a[0])*2,decimals=0))
        self._nbmax[1] = int(np.around(np.linalg.norm(self._a[1])*2,decimals=0))
        self._nbmax[2] = int(np.around(np.linalg.norm(self._a[2])*2,decimals=0))
        #print(self._nbmax)


        gpoints = []
        G_ind = 0
        i3,j2,k1 = np.mgrid[-self._nbmax[2]:self._nbmax[2]:(2 * self._nbmax[2] + 1)*1j,-self._nbmax[1]:self._nbmax[1]:(2 * self._nbmax[1] + 1)*1j,
                   -self._nbmax[0]:self._nbmax[0]:(2 * self._nbmax[0] + 1) * 1j]
        i3 = i3.flatten()
        j2 = j2.flatten()
        k1 = k1.flatten()
        #j2 = np.arange(2 * self._nbmax[1] + 1)- self._nbmax[1]
        #k1 = np.arange(2 * self._nbmax[0] + 1)- self._nbmax[0]

        #for i in range(2 * self._nbmax[2] + 1):
        #    i3 = i - self._nbmax[2] # - 2 * self._nbmax[2] - 1 if i > self._nbmax[2] else i
        #    for j in range(2 * self._nbmax[1] + 1):
        #        j2 = j - self._nbmax[1] # - 2 * self._nbmax[1] - 1 if j > self._nbmax[1] else j
        #        for k in range(2 * self._nbmax[0] + 1):
        #            k1 = k - self._nbmax[0] # - 2 * self._nbmax[0] - 1 if k > self._nbmax[0] else k
        G = np.array([k1, j2, i3])
        v = np.array(np.array([kpt]).T + G).T
        g = np.linalg.norm(np.dot(v, self.b),axis=1)
        E = np.array( g ** 2 / self._C )
        #if E < self.encut:
        #    gpoints.append(G)
        #    G_ind += 1
        gpoints1 = G[0][E<self.encut]
        gpoints2 = G[1][E<self.encut]
        gpoints3 = G[2][E<self.encut]
        gpoints = np.array([gpoints1,gpoints2,gpoints3]).T
        return gpoints

    def get_overlap_matrix(self,orbitalWF,secondWF = None):
        if secondWF == None:
            secondWF = orbitalWF
        num_orbs = len(orbitalWF)
        num_orbs2 = len(secondWF)
        #grid = orbitalWF[list(orbitalWF.keys())[0]].shape
        #print(num_orbs)
        overlap_matrix = np.zeros((num_orbs, num_orbs2), dtype=np.complex_)
        #for_integral = np.zeros((num_orbs,num_orbs, grid[0], grid[1], grid[2]), dtype=np.complex_)
        for orb1 in range(num_orbs):
            WF1 = orbitalWF[list(orbitalWF.keys())[orb1]]
            for orb2 in range(num_orbs2):
                WF2 = secondWF[list(secondWF.keys())[orb2]]
                overlap_matrix[orb1][orb2] = reciprocal_integral(np.conj(WF1) * WF2,self.cell_vol)
        return overlap_matrix

    def get_ham(self,kpt,return_overlap=False,return_truevec = True):
        ham = np.zeros((self.num_orbs,self.num_orbs),dtype=np.complex_)
        kdep_overlap = np.zeros((self.num_orbs,self.num_orbs),dtype=np.complex_)
        vec_to_orbs = np.zeros(self.vec_to_trans.shape)
        #print("check same:", self.orbitals[str(0)][12,12,:])
        
        #vecs = np.array(self.vec_to_trans)
        #orb_pos = np.array(self.orb_redcoords)
        #vec_to_orbs = vecs[None,None,:,:,:] + orb_pos[None,:,None,None,None] - orb_pos[:,None,None,None,None] #[orb1,orb2,t1,t2,t3]
        #print("shape of vec:",vec_to_orbs.shape)
        #hold_vecs = np.reshape(vec_to_orbs,(self.num_orbs*self.num_orbs*self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
        #cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        #dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        if not hasattr(self, 'flat_vec_to_orbs'):
            vecs = np.array(self.vec_to_trans)
            orb_pos = np.array(self.orb_redcoords)
            vec_to_orbs = vecs[None,None,:,:,:] + orb_pos[None,:,None,None,None] - orb_pos[:,None,None,None,None] #[orb1,orb2,t1,t2,t3]
            #print("shape of vec:",vec_to_orbs.shape)
            hold_vecs = np.reshape(vec_to_orbs,(self.num_orbs*self.num_orbs*self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
            cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
            dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
            self.dist_to_orbs = dist_to_orbs
            self.flat_vec_to_orbs = hold_vecs
        
        exp_fac = np.exp(2j*np.pi*np.dot(kpt,self.flat_vec_to_orbs))
        overlap_fac = copy.deepcopy(exp_fac)
        if self.set_min == True:
            overlap_fac[self.dist_to_orbs > self.min_overlap_dist] = 0
        overlap_fac = np.reshape(overlap_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        kdep_overlap = np.sum(self.overlaps_params * overlap_fac,axis=(2,3,4))
        
        if self.set_min == True:
            exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
        exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        ham = np.sum(self.TB_params*exp_fac,axis=(2,3,4))
        
        if self.orbs_orth == True:
            (eigval, eigvec) = np.linalg.eigh(ham)
            true_eigvec = eigvec
        else:
            kdep_Sij = kdep_overlap
            #self.Sij[kpt] = kdep_Sij
            #print("check overlap:",kdep_Sij)

            eigenvalj, kdep_Dij = np.linalg.eigh(kdep_Sij)
            # check correctness of eigen
            # construct Aij
            kdep_Aij = np.zeros((self.num_orbs, self.num_orbs), dtype=np.complex_)
            for j in range(self.num_orbs):
                kdep_Aij[:, j] = kdep_Dij[:, j] / (eigenvalj[j]) ** (1 / 2)
            #get ham for orthogonal eigenvectors
            conj_Aij = np.conj(kdep_Aij).transpose()
            AtHA = np.matmul(conj_Aij,np.matmul(ham,kdep_Aij))
            AtHA = (AtHA + np.conj(AtHA).T)/2 # ensure that the hamiltonian is perfectly Hermitian
            #ham = np.matmul(np.linalg.inv(conj_Aij),np.matmul(AtHA,np.linalg.inv(kdep_Aij)))
            eigval,eigvec = np.linalg.eigh(AtHA)
            if return_truevec == True:
                true_eigvec = np.matmul(kdep_Aij,eigvec)
            else:
                true_eigvec = eigvec
                
            # test that you can reconstruct the hamiltonian from eigenvecs
            #test_AtHA = np.matmul(eigvec,np.matmul(np.diag(eigval), np.linalg.inv(eigvec)))
            ##newAtHA = (AtHA + np.conj(AtHA).T)/2
            #diff_AtHA = AtHA - test_AtHA
            #if (diff_AtHA.flatten() > 0.0001).any():
            #    print("couldn't construct the same ham!",diff_AtHA.flatten()[diff_AtHA.flatten() > 0.00001])

        if return_overlap == True and self.orbs_orth == False:
            return eigval, true_eigvec, kdep_Sij#Aij, ham
        else:
            return eigval,true_eigvec

    '''
    will update later to include converting the recip orb to real orbs and plotting
    def plot_sphOrbitals(self):
        figures = {}
        axises = {}

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
    
    def get_uniform(self,grid):
        #print(grid)
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        # creating the k point grid - no weights b/c it is not the reduced form of the kpoint grid
        structure = Structure.from_file(self.directory + 'POSCAR')
        # pass "structure" to define class into "space"
        space = SpacegroupAnalyzer(structure)
        #print("vecs:",self._b)
        #print(space._structure.lattice.reciprocal_lattice)
        # call the function within the class
        mesh,maps = space.get_ir_reciprocal_mesh_map(grid)
        
        py_ir_kpts_inf = space.get_ir_reciprocal_mesh(grid)
        py_ir_kpts = np.zeros((len(py_ir_kpts_inf),3))
        py_ir_count = np.zeros((len(py_ir_kpts_inf)))
        for ind in range(len(py_ir_kpts_inf)):
            py_ir_kpts[ind] = py_ir_kpts_inf[ind][0]
            py_ir_count[ind] = py_ir_kpts_inf[ind][1]
        cart_mesh = _red_to_cart((self._b[0], self._b[1], self._b[2]), mesh)
        #print("cart kpts",cart_mesh)
        irrec_kpts_ind,irrec_kpts_count = np.unique(maps,return_counts=True)
        irrec_kpts = np.array(mesh)[irrec_kpts_ind]
        #print("irreducible kpts:",irrec_kpts)
        kptvecs = mesh
        kptvecs = np.array(kptvecs, dtype=np.float_)
        #print(kptvecs)
        num_kpts = len(kptvecs)
        self.num_kpts = num_kpts
        self.kpoints = kptvecs
        self.kpt_weights = np.ones(num_kpts)/num_kpts
        
        eigvals = np.zeros((num_kpts,self.num_orbs))
        eigvecs = np.zeros((num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex_)
        kdep_Sij = np.zeros((num_kpts, self.num_orbs, self.num_orbs), dtype=np.complex_)  # from get_ham
        
        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)
        vec_to_orbs = vecs[None,None,:,:,:] + orb_pos[None,:,None,None,None] - orb_pos[:,None,None,None,None] #[orb1,orb2,t1,t2,t3]
        #print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs,(self.num_orbs*self.num_orbs*self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        self.dist_to_orbs = dist_to_orbs
        self.flat_vec_to_orbs = hold_vecs
        
        for kpt in range(num_kpts):
            pass
                #print("-",end="")
        #print("")
        for kpt in range(num_kpts):
            #print("-",end="")
            eigvals[kpt],eigvecs[kpt],kdep_Sij[kpt] = self.get_ham(kptvecs[kpt],return_overlap=True)
        #print("")
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.Sij = kdep_Sij
        
        
        #def get_partial_charge(self):
        N_i = np.zeros(self.num_orbs)
        mnk2 = np.transpose(eigvecs,axes=(0,2,1)) #[kpt,band,orb]
        cj = mnk2[:, :, None, :]  # self.mnkcoefficients[:,:,None,:]
        ci2 = np.conj(mnk2[:, :, :, None])  # self.mnkcoefficients[:,:,:,None])
        Sij = kdep_Sij[:, None, :, :]

        fnk = np.zeros((num_kpts, self.num_orbs), dtype=np.float_)
        for k in range(num_kpts):
            for i in range(self.num_orbs):
                if eigvals[k, i] <= (self.efermi+self.energy_shift + 0.0001):
                    fnk[k, i] = 1
                elif eigvals[k, i] > (self.efermi+self.energy_shift + 0.0001):
                    fnk[k, i] = 0
        new_fnk = fnk[:, :, None, None]
        Ni_value = ci2 * cj * new_fnk * Sij / num_kpts
        sum_k = np.sum(Ni_value, axis=0)
        sum_n = np.sum(sum_k, axis=0)
        sum_j = np.sum(sum_n, axis=0)
        N_i = sum_j
        realN_i = np.real(N_i)
        #print('realNi: ', realN_i)
        sum_Ni = np.sum(realN_i)
        #print('sum: ', sum_Ni)
        
        # get just the atomic portion (no charge from the overlap density)
        diag_Sij = kdep_Sij.diagonal(0,1,2) # [kpt,orb]
        #print("diag Sij:",diag_Sij)
        ci_2 = np.conj(mnk2)*mnk2 #[kpt,band,orb]
        get_atomic_Ni = ci_2*diag_Sij[:,None,:]*fnk[:,:,None] / num_kpts
        sum_k = np.sum(get_atomic_Ni, axis=0)
        atomic_Ni = np.sum(sum_k, axis=0).real
        #print("atomic Ni:",atomic_Ni)

        # output charge on each atom
        # write num protons into tb_input.txt
        N_e = []
        for each_atom in range(max(self.orbatomnum) + 1):
            num_e_each = 0  # starts at 0, resets for every atom
            for i in range(len(self.orbatomnum)):
                if self.orbatomnum[i] == each_atom:
                    num_e_each = realN_i[i] + num_e_each
            N_e.append(num_e_each)

        #print("num_pro: ", self.num_pro)
        #print('unique elements: ', self.unique_elems)

        #N_p = np.zeros(len(self.elements))
        # print('beginingn Ne array: ', N_p_current)
        #for i in range(len(self.elements)):
        #    one_atom = self.elements[i]
        #    # print(one_atom)
        #    ind = np.where(np.array(self.unique_elems) == one_atom)[0][0]
        #    # print(ind)
        #    N_p[i] = self.num_pro[ind]
        #print('corrected Np array based on atoms in the system:', N_p)

        # num_e_each = 2 * num_e_each # multiply by 2, since Ni doesn't account for degeneracy
        N_e = np.array(N_e) * 2
        #print('The electron occupation for the atoms ', self.elements, ' is ', N_e)

        #oxid_state = N_p - N_e
        #print('the oxidation state for the atoms ', self.elements, ' is ', oxid_state)

    def read_mnkcoeff(self):
        #check = exists(self.directory+"mnkcoeff.txt")
        #if check == False:
        #    print("SORRY, YOU DO NOT HAVE THE NECESSARY FILE IN THIS FOLDER")
        #    print("To generate the file run get_orbsandcoeffs(), converge_orbs(), and write_mnkcoeff()")
        mnkfile = open(self.directory+"mnkcoeff.txt","r")
        lines = mnkfile.readlines()
        #print(lines[1])
        [numbands,numkpts,numorbs] = [int(i) for i in lines[1].strip().split()]
        self.max_band = numbands
        #print(numbands,numkpts,numorbs)
        self.mnkcoefficients = np.zeros((numkpts,numbands,numorbs), dtype=np.complex_) #kpt, band,orb
        #print(self.mnkcoefficients.shape)
        for line in lines[2:]:
            [band,orb,kpt] = [int(i)-1 for i in line.strip().split()[:3]]
            #print(band,kpt,orb)
            real = float(line.strip().split()[3])
            imag = float(line.strip().split()[4])
            self.mnkcoefficients[kpt,band,orb] = real + imag*1j
        allspillage = []
        allavg_spillage = []
        for kpt in range(numkpts):
            sqr_coeff = np.square(np.real(self.mnkcoefficients[kpt])) + np.square(np.imag(self.mnkcoefficients[kpt]))
            spillage = np.sum(sqr_coeff[:self.num_orbs, :], axis=1)  # +np.sum(sqr_coeff[:8,8:12], axis=1)
            allspillage.append(spillage)
            #print("kpt",self.kpoints[kpt],np.average(spillage))
            #print("spill:", spillage)
            allavg_spillage.append(np.average(spillage))
        #print("average spillage (possibly not normed)", np.average(allavg_spillage))

    def get_neighbors(self):
        #vecs = np.array(self.vec_to_trans)
        #orb_pos = np.array(self.orbpos)
        #vec_to_orbs = vecs[:,:,:,None,None] + orb_pos[None,None,None,None,:] - orb_pos[None,None,None,:,None]
        #print("shape of vec:",vec_to_orbs.shape)
        #hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2]*self.num_orbs*self.num_orbs,3)).transpose()
        #cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        #dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        dist_to_orbs = self.dist_to_orbs
        
        small_dist_bool = dist_to_orbs < 10
        small_dist_index = np.arange(len(dist_to_orbs))[small_dist_bool]
        small_dist_orbs = dist_to_orbs[small_dist_index]
        sorted_dist_orbs = np.around(np.sort(small_dist_orbs),decimals=4)
        NN_dists = np.unique(sorted_dist_orbs)
        NN_index = {}
        for NN in range(min(10,len(NN_dists))):
            small_indices = np.abs(small_dist_orbs - NN_dists[NN]) <= 0.001
            NN_index[NN] = small_dist_index[small_indices] #flattened indices
        self.NN_index = NN_index
        self.NN_dists = NN_dists[:10]

    def make_COHP_dashapp(self):
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State

        # Sample data
        keys = np.unique(self.elements)
        data = {}
        for elem in keys:
            elemorbs = np.array(self.exactorbtype, dtype='<U1')[np.array(self.orbatomname) == elem]
            data[elem] = list(np.unique(elemorbs)[::-1])
        #print(data)
        #data = {"Si": ["s", "p"], "Pb": ["s", "p", "d"]}

        # Initialize Dash app
        app = dash.Dash(__name__)

        # Define a recursive function to create nested checklist items
        def create_nested_checklist(data, level=0,orb=0):
            checklist_items = []
            for atom, orbitals in data.items():
                label = atom if level == 0 else f"{atom} ({', '.join(orbitals)})"
                checklist_items.append(html.Div([
                                                   label,
                                                   dcc.Checklist(
                                                       id={'type': 'orbitals-checklist'+str(orb), 'index': atom},
                                                       options=[{'label':"   "+orbital, 'value': f"{atom}-{orbital}"} for
                                                                orbital in orbitals],
                                                       value=[f"{atom}-{orbital}" for orbital in orbitals],
                                                       inline=True,labelStyle= {"width":"3rem"}
                                                   ),
                                               ] + create_nested_checklist({}, level + 1)))
            return checklist_items

        # App layout
        NNs = np.append(["All"],range(1,10))
        #print(NNs)
        app.layout = html.Div([html.Div([html.H1("Orbital \u03B1", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                        html.Ul(create_nested_checklist(data,orb=0),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Orbital \u03B2", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                      html.Ul(create_nested_checklist(data,orb=1),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Nearest Neighbor", style={'margin': '5px', 'text-align': 'center', 'font-size': 20,'height':20,'width':100,'display': 'inline-block'}),
                          dcc.Dropdown(id='nearest-neighbor', options=[{'label': str(option), 'value': option} for option in NNs],value=NNs[0],style={'margin':'5px',"width":60,'display': 'inline-block'})],
                         style={"width":180,'display': 'inline-block'}),
            html.Button('Calculate new COHP', id='print-button', n_clicks=0),
            html.Div(dcc.Graph(id='COHP-figure')),
            dcc.Store(id='selected-items-store', data=[data,data])
            ],style={'height': 500,'width':600})

        # Callbacks to dynamically generate the selected items callback for each atom
        for atom in list(data.keys()):
            @app.callback(
                [Output({'type': 'orbitals-checklist0', 'index': atom}, 'value'),
                 Output('selected-items-store', 'data', allow_duplicate=True)],
                [Input({'type': 'orbitals-checklist0', 'index': atom}, 'value')],
                [State('selected-items-store', 'data')],
                prevent_initial_call=True
            )
            def update_selected_items(selected_items, stored_data, atom=atom):
                stored_data[0][atom] = [item.split('-')[1] for item in selected_items]
                return selected_items, stored_data

        # Callbacks to dynamically generate the selected items callback for each atom
        for atom in list(data.keys()):
            @app.callback(
                [Output({'type': 'orbitals-checklist1', 'index': atom}, 'value'),
                 Output('selected-items-store', 'data', allow_duplicate=True)],
                [Input({'type': 'orbitals-checklist1', 'index': atom}, 'value')],
                [State('selected-items-store', 'data')],
                prevent_initial_call=True
            )
            def update_selected_items(selected_items, stored_data, atom=atom):
                stored_data[1][atom] = [item.split('-')[1] for item in selected_items]
                return selected_items, stored_data

        # Callback to handle print button click and print selected items
        @app.callback(
            Output('COHP-figure', 'figure'),
            [Input('print-button', 'n_clicks')],
            [State('selected-items-store', 'data'),State('nearest-neighbor','value')],

        )
        def print_selected_items(n_clicks, stored_data,NN):
            #print("orb for COHP",stored_data)
            fig = self.get_COHP("BS",orbs=stored_data,colorhalf=15,NN=NN)
            return fig

        # Run the app
        #if __name__ == '__main__':
        app.run_server(debug=True)

    def get_COHP(self,mode,orbs,NN=None,sigma=0.1,ylim=None,colorhalf=None,include_onsite=False,point=None):
        '''
        :param mode: string; "point": want COHP at specific band and kpoint
                            "DOS": want COHP integrated over reciprocal space to produce DOS
                            "total": integrate of reciprocal space and energy to get total interaction (recommend setting onlyOccupied=True)
        :param orbs: array; either gives elements and orbital type (eg [[["Pb"],["p"]],[["O"],["s"]]]) or give list of orb numbers [[1,2],[3,5]]
        :param point: array; first element is kpoint index, second is band index (how these are sorted is determined by other functions, may want to disentangle bands before hand)
        :param onlyOccupied: only for mode = "DOS" or "total", only integrate over occupied states (not very relavent for "DOS", but may save time)
        :param energy_range: only for mode = "DOS" or "total", only integrate over states in energy range (not very relavent for "DOS", but may save time)
        :return: float ("point" or "total") or 2D array ("DOS", x=COHP,y=E) of COHP values of either the point, energy, or total
        '''


        if ylim == None:
            minen = np.amin(self.evals.flatten() - (self.efermi+self.energy_shift))
            maxen = np.amax(self.evals.flatten() - (self.efermi+self.energy_shift))
            ylim = (minen-0.5,maxen+0.5)

        if type(orbs[0]) is dict:
            orbs_dict = orbs
            firstorbs = []
            secorbs = []
            #format of orbs should be {"Si":["s","p"],"Pb":["s"]}
            # find orbs for the element
            for element in orbs[0].keys():
                #print(element)
                for orbtype in orbs[0][element]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            firstorbs.append(addorb)

            for element in orbs[1].keys():
                for orbtype in orbs[1][element]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            secorbs.append(addorb)
            orbs = [firstorbs,secorbs]
            #print(orbs)
        
        #reference indices based on the nearest neighbors, much faster and can look at specific NN contributions
        num_each_dir = self.num_each_dir
        if not hasattr(self, 'NN_index'):
                self.get_neighbors()
        if NN == None or NN == "All":
            indices = np.array([],dtype=np.int_)
            for nn in range(len(self.NN_dists)):
                indices = np.append(indices,self.NN_index[nn])
            total_index = indices
            #print("NN distances that are included:",self.NN_dists)
        else:
            NN = int(NN)
            if NN > len(self.NN_dists):
                #print("give a lower NN integer!")
                pass
            NN_dist = self.NN_dists[int(NN)]
            total_index = self.NN_index[int(NN)]
            #print("NN number, distance, count, and indices:", int(NN),NN_dist,len(total_index))#,NN_index)
        
        #convert the flattens indices into the full matrix indices
        full_indices = np.unravel_index(total_index,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        #only select terms which have the orbitals the user is seeking
        orb1_in_full = full_indices[0][None,:]==np.array([orbs[0]]).T
        orb2_in_full = full_indices[1][None,:]==np.array([orbs[1]]).T
        in_full1 = np.full(len(orb1_in_full[0]),False)
        in_full2 = np.full(len(orb2_in_full[0]),False)
        for bool_list in orb1_in_full:
            in_full1 = in_full1 | bool_list
        for bool_list in orb2_in_full:
            in_full2 = in_full2 | bool_list
        in_full = in_full1 & in_full2
        #remove onsite terms
        is_onsite = (full_indices[2] == num_each_dir[0]) & (full_indices[3] == num_each_dir[1]) & (full_indices[4] == num_each_dir[2]) & (full_indices[0]==full_indices[1])
        #print(in_full)
        if include_onsite == False:
            in_full = in_full & np.logical_not(is_onsite)
        #print(in_full)
        full_ind = [full_indices[0][in_full],full_indices[1][in_full],full_indices[2][in_full],full_indices[3][in_full],full_indices[4][in_full]]
        #print("num of params:", len(full_ind[0]))

        # generate the hamiltonian matrix for neccessary kpoints
        if mode == "point":
            kpoint = point[0]
            ham_nmk = self.hamilton[:,:,kpoint]#self.get_hamiltonian(kpoints=kpoint)
        #else:
        #    ham_nmk = self.hamilton#self.get_hamiltonian()

        # get the COHP values
        if mode == "point":
            band = point[1]
            cohp = 0
            for orb1 in orbs[0]:
                for orb2 in orbs[1]:
                    cohp += np.conj(self.eigvecs[kpoint,band,orb1])*ham_nmk[orb1,orb2]*self.mnkcoefficients[kpoint,band,orb2]

        if mode == "DOS" or "BS":
            cohp = np.zeros(self.eigvecs[:,:self.num_orbs,0].shape, dtype=np.complex_) # kpts,bands
            
            #get it from TB params so have all except orbital energies
            vecs = np.array(self.vec_to_trans)
            orb_pos = np.array(self.orb_redcoords)

            energysum = 0
            overlapsum = 0
            for kind in range(self.num_kpts):
                kpoint = self.kpoints[kind]
                '''
                # use all the parameters
                exp_fac = np.exp(2j*np.pi*np.dot(kpoint,self.flat_vec_to_orbs))
                if self.set_min == True:
                    exp_fac[self.dist_to_orbs > self.min_hopping_dist] = 0
                exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
                ham = np.sum(self.TB_params*exp_fac,axis=(2,3,4)) # [orb1,orb2]
                coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
                energyCont = ham[None,:,:]*np.conj(coeff[:,:,None])*coeff[:,None,:] # [band,orb1,orb2]
                cohp[kind] = np.sum(energyCont,axis=(1,2)) * self.kpt_weights[kind]
                '''
                #now only use the indices that are short
                vec_to_orbs = vecs[full_ind[2],full_ind[3],full_ind[4]] + orb_pos[full_ind[1]] - orb_pos[full_ind[0]]
                exp_fac = np.exp(2j*np.pi*np.dot(kpoint,vec_to_orbs.T))
                scaledTB = self.TB_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
                #scaledOvlap = self.overlaps_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
                coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
                #coeff = np.transpose(coeff,axes = (0,2,1))
                energyCont = scaledTB[None,:]*np.conj(coeff)[:,full_ind[0]]*coeff[:,full_ind[1]]
                partial_enCont = energyCont
                cohp[kind] = np.sum(partial_enCont,axis=1) * self.kpt_weights[kind]
                
            integrated_cohp = np.sum(cohp.flatten()[self.eigvals[:, :].flatten()<self.efermi+self.energy_shift])
            #print("integrated occupied cohp:",integrated_cohp)
            #print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
            shifted_energies = self.eigvals[:, :].flatten()-(self.efermi+self.energy_shift)
            
            if mode == "DOS":
                fig = plt.figure()
                fig.set_size_inches(3, 5)
                ax = fig.add_subplot(111)
                numbins = 100
                flat_cohp = cohp.flatten()

                # create the DOS by histograms
                #myproj, x, _ = ax.hist(shifted_energies[shifted_energies < 1], bins=numbins,weights=flat_cohp[shifted_energies < 1],color="white", orientation='horizontal')#- self.efermi
                #smooth_cohp = smooth(myproj, 2)
                #smooth_cohp = myproj

                # create the DOS by sum of gaussians
                flat_cohp = flat_cohp[shifted_energies < ylim[1]]
                shifted_energies = shifted_energies[shifted_energies < ylim[1]]
                minx = np.min(shifted_energies)-sigma*2
                maxx = np.max(shifted_energies)#+sigma*2
                points = 200
                energies = np.linspace(minx,maxx,points)
                sig = sigma#0.15#5/self.num_kpts**(1/2)
                all_cohp = 1/(sig*(2*np.pi)**(1/2))*np.exp(-1/2*((energies[:,None]-shifted_energies[None,:])/sig)**2)*flat_cohp[None,:]
                total_cohp = np.sum(all_cohp,axis=1)

                #ax.plot(x[:-1],np.zeros(len(x[:-1])),"--",color="gray")
                #ax.plot(x[:-1], smooth_cohp, color="black")
                #ax.plot([0,0],[np.min(smooth_cohp)-0.1,np.max(smooth_cohp)+0.1],color="black")
                #plot vertically
                #ax.plot(np.zeros(len(x[:-1])),x[:-1],"--",color="gray")
                #ax.plot(smooth_cohp,x[:-1], color="black")
                ax.plot(np.zeros(points),energies,"--",color="gray")
                ax.plot(total_cohp,energies,color="black")
                mini = np.min(total_cohp)-0.1
                maxi = np.max(total_cohp)+0.1
                #mini = -1.2
                #maxi = 0.5
                ax.plot([mini,maxi],[0,0],"--",color="gray")
                ax.set_ylabel("Energy (eV)")
                ax.set_xlabel("COHP")
                plt.ylim(ylim)
                #print("showing plot")
                #plt.xlim((mini,maxi))
                plt.gca().invert_xaxis()
                fig.tight_layout()
                plt.show()
            elif mode == "BS":
                return self.plotBS(color_label = "COHP",colors=cohp.real*self.num_kpts,colorhalf=colorhalf,ylim=ylim)

    def get_COOP(self,mode,orbs,point=None,energy_range=None,NN=None,sigma=0.1):
        '''
        :param mode: string; "point": want COOP at specific band and kpoint
                            "DOS": want COOP integrated over reciprocal space to produce DOS
                            "total": integrate of reciprocal space and energy to get total interaction (recommend setting onlyOccupied=True)
        :param orbs: array; either gives elements and orbital type (eg [[["Pb"],["p"]],[["O"],["s"]]]) or give list of orb numbers [[1,2],[3,5]]
        :param point: array; first element is kpoint index, second is band index (how these are sorted is determined by other functions, may want to disentangle bands before hand)
        :param onlyOccupied: only for mode = "DOS" or "total", only integrate over occupied states (not very relavent for "DOS", but may save time)
        :param energy_range: only for mode = "DOS" or "total", only integrate over states in energy range (not very relavent for "DOS", but may save time)
        
        :return: float ("point" or "total") or 2D array ("DOS", x=COOP,y=E) of COOP values of either the point, energy, or total
        '''
        if type(orbs[0][0]) is not int:
            firstorbs = []
            secorbs = []
            # find orbs for the element
            for element in orbs[0][0]:
                for orbtype in orbs[0][1]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            firstorbs.append(addorb)
            for element in orbs[1][0]:
                for orbtype in orbs[1][1]:
                    for addorb in range(self.num_orbs):
                        if self.orbatomname[addorb] == element and orbtype in self.exactorbtype[addorb]:
                            secorbs.append(addorb)
            orbs = [firstorbs,secorbs]
            #print(orbs)
            
        #reference indices based on the nearest neighbors, much faster and can look at specific NN contributions
        num_each_dir = self.num_each_dir
        if not hasattr(self, 'NN_index'):
                self.get_neighbors()
        if NN == None:
            indices = np.array([],dtype=np.int_)
            for nn in range(len(self.NN_dists)):
                indices = np.append(indices,self.NN_index[nn])
            total_index = indices
            #print("NN distances that are included:",self.NN_dists)
        else:
            if NN > len(self.NN_dists):
                #print("give a lower NN integer!")
                pass
            NN_dist = self.NN_dists[int(NN)]
            total_index = self.NN_index[int(NN)]
            #print("NN number, distance, count, and indices:", int(NN),total_index,len(total_index))#,NN_index)
        
        #convert the flattens indices into the full matrix indices
        full_indices = np.unravel_index(total_index,(self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        #only select terms which have the orbitals the user is seeking
        orb1_in_full = full_indices[0][None,:]==np.array([orbs[0]]).T
        orb2_in_full = full_indices[1][None,:]==np.array([orbs[1]]).T
        in_full1 = np.full(len(orb1_in_full[0]),False)
        in_full2 = np.full(len(orb2_in_full[0]),False)
        for bool_list in orb1_in_full:
            in_full1 = in_full1 | bool_list
        for bool_list in orb2_in_full:
            in_full2 = in_full2 | bool_list
        in_full = in_full1 & in_full2
        #remove onsite terms
        is_onsite = (full_indices[2] == num_each_dir[0]) & (full_indices[3] == num_each_dir[1]) & (full_indices[4] == num_each_dir[2]) & (full_indices[0]==full_indices[1])
        #print(in_full)
        in_full = in_full & np.logical_not(is_onsite)
        #print(in_full)
        full_ind = [full_indices[0][in_full],full_indices[1][in_full],full_indices[2][in_full],full_indices[3][in_full],full_indices[4][in_full]]
        #print("num of params:", len(full_ind[0]))
        
        
        if mode == "DOS" or "BS":
            coop = np.zeros(self.eigvecs[:,:self.num_orbs,0].shape, dtype=np.complex_) # kpts,bands
            
            #get it from TB params so have all except orbital energies
            vecs = np.array(self.vec_to_trans)
            orb_pos = np.array(self.orb_redcoords)

            energysum = 0
            overlapsum = 0
            for kind in range(self.num_kpts):
                kpoint = self.kpoints[kind]
                #now only use the indices that are short
                vec_to_orbs = vecs[full_ind[2],full_ind[3],full_ind[4]] + orb_pos[full_ind[1]] - orb_pos[full_ind[0]]
                exp_fac = np.exp(2j*np.pi*np.dot(kpoint,vec_to_orbs.T))
                #scaledTB = self.TB_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
                scaledOvlap = self.overlaps_params[full_ind[0],full_ind[1],full_ind[2],full_ind[3],full_ind[4]]*exp_fac
                coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
                #coeff = np.transpose(coeff,axes = (0,2,1))
                energyCont = scaledOvlap[None,:]*np.conj(coeff)[:,full_ind[0]]*coeff[:,full_ind[1]]
                partial_enCont = energyCont
                coop[kind] = np.sum(partial_enCont,axis=1) * self.kpt_weights[kind]
            integrated_coop = np.sum(coop.flatten()[self.eigvals[:, :].flatten()<self.efermi+self.energy_shift])
            #print("integrated occupied coop:",integrated_coop)
            fig = plt.figure()
            fig.set_size_inches(3, 5)
            ax = fig.add_subplot(111)
            numbins = 100
            #print("max eigval:",self.eigval[:, self.num_orbs-1, 0])
            # get by histogram
            #myproj, x, _ = ax.hist(self.eigvals[:, :].flatten()-(self.efermi+self.energy_shift), bins=numbins,weights=coop.flatten())#- self.efermi
            #smooth_coop = smooth(myproj, 2)
            #ax.plot(x[:-1], smooth_coop, color="black")
            
            # get by sum of gaussians
            shifted_energies = self.eigvals[:, :].flatten()-(self.efermi+self.energy_shift)
            flat_coop = coop.flatten()
            
            flat_coop = flat_coop[shifted_energies < 2]
            shifted_energies = shifted_energies[shifted_energies < 2]
            minx = np.min(shifted_energies)-0.1
            maxx = np.max(shifted_energies)+0.1
            points = 200
            energies = np.linspace(minx,maxx,points)
            sig = sigma#0.15#5/self.num_kpts**(1/2)
            all_coop = 1/(sig*(2*np.pi)**(1/2))*np.exp(-1/2*((energies[:,None]-shifted_energies[None,:])/sig)**2)*flat_coop[None,:]
            total_coop = np.sum(all_coop,axis=1)
            
            ax.plot(np.zeros(points),energies,"--",color="gray")
            ax.plot(total_coop,energies,color="black")
            mini = np.min(total_coop)-0.1
            maxi = np.max(total_coop)+0.1
            ax.plot([mini,maxi],[0,0],"--",color="gray")
            ax.set_ylabel("Energy (eV)")
            ax.set_xlabel("COOP")
            #plt.ylim((-3,4))
            #print("showing plot")
            #plt.xlim((mini,maxi))
            plt.gca().invert_xaxis()
            fig.tight_layout()
            
            #plt.ylim((-3,4))
            #print("showing plot")
            plt.show()

    def get_bonds_figure(self):
        # this will plot the crystal structure atoms with line weighted by iCOHP 
        # each line should also be hoverable to reveal the number and amounts that are s-s,s-p, and p-p
        # there should also be a COHP summary in the corner giving iCOHP over the 1NN, 2NN, etc
        import plotly.graph_objs as go
        if not hasattr(self, 'NN_index'):
                self.get_neighbors()
        #atom size based on element
        all_atom_rad = {"H":0.53,"He":0.31,"Li":1.51,"Be":1.12,"B":0.87,"C":0.67,"N":0.56,"O":0.48,"F":0.42,"Ne":0.38,
                    "Na":1.90,"Mg":1.45,"Al":1.18,"Si":1.11,"P":0.98,"S":0.88,"Cl":0.79,"Ar":0.71,
                    "K":2.43,"Ca":1.94,"Sc":1.84,"Ti":1.76,"V":1.71,"Cr":1.66,"Mn":1.61,"Fe":1.56,"Co":1.52,
                    "Ni":1.49,"Cu":1.45,"Zn":1.42,"Ga":1.36,"Ge":1.25,"As":1.14,"Se":1.03,"Br":0.94,"Kr":0.88,
                    "Rb":2.65,"Sr":2.19,"Y":2.12,"Zr":2.06,"Nb":1.98,"Mo":1.90,"Tc":1.83,"Ru":1.78,"Rh":1.73,
                    "Pd":1.69,"Ag":1.65,"Cd":1.61,"In":1.56,"Sn":1.45,"Sb":1.33,"Te":1.23,"I":1.15,"Xe":1.08,
                    "Cs":2.98,"Ba":2.53,"Lu":2.17,"Hf":2.08,"Ta":2.00,"W":1.93,"Re":1.88,"Os":1.85,"Ir":1.80,
                    "Pt":1.77,"Au":1.75,"Hg":1.71,"Tl":1.56,"Pb":1.54,"Bi":1.43,"Po":1.35,"At":1.27,"Rn":1.20}
        
        each_dir = np.array([1,1,1])
        cell = each_dir*2+1
        vec_to_trans = np.zeros((cell[0],cell[1],cell[2],3))
        for x in range(cell[0]):
            for y in range(cell[1]):
                for z in range(cell[2]):
                    vec_to_trans[x,y,z] = [x-each_dir[0],y-each_dir[1],z-each_dir[2]]
                    
        num_total_atoms = self.numAtoms*cell[0]*cell[1]*cell[2]
        vec_to_atoms = vec_to_trans[None,:,:,:]+np.array(self.primAtoms)[:,None,None,None]
        vec_to_atoms = np.reshape(vec_to_atoms,(num_total_atoms,3)).T
        #print(vec_to_atoms)
        cart_to_atoms = _red_to_cart((self._a[0], self._a[1], self._a[2]), vec_to_atoms.transpose()).T
        
        # plot the atoms onto the crystal
        x_data = cart_to_atoms[0]
        y_data = cart_to_atoms[1]
        z_data = cart_to_atoms[2]
        #crystal = go.Scatter3d(x=x_data,y=y_data,z=z_data,mode='markers',marker=dict(size=20,color='blue',opacity=0.8),hoverinfo='none')
        
        
        # get the cohp
        cohp = np.zeros((self.num_kpts,self.num_orbs,self.num_orbs,cell[0],cell[1],cell[2]), dtype=np.complex_) # kpts,orb1,orb2,T1,T2,T3
        #get it from TB params so have all except orbital energies
        vecs = np.array(vec_to_trans) # this will align different with the TB_params!!
        orb_pos = np.array(self.orb_redcoords)
        param_dir = self.num_each_dir
        energysum = 0
        overlapsum = 0
        for kind in range(self.num_kpts):
            kpoint = self.kpoints[kind]
            #now only use the indices that are short
            vec_to_orbs = vecs[None,None,:,:,:] + orb_pos[None,:,None,None,None] - orb_pos[:,None,None,None,None]
            hold_vecs = np.reshape(vec_to_orbs,(self.num_orbs*self.num_orbs*cell[0]*cell[1]*cell[2],3))
            exp_fac = np.exp(2j*np.pi*np.dot(kpoint,hold_vecs.T))
            exp_fac = np.reshape(exp_fac,(self.num_orbs,self.num_orbs,cell[0],cell[1],cell[2]))
            
            scaledTB = self.TB_params[:,:,param_dir[0]-each_dir[0]:param_dir[0]+each_dir[0]+1,
                                      param_dir[1]-each_dir[1]:param_dir[1]+each_dir[1]+1,
                                      param_dir[2]-each_dir[2]:param_dir[2]+each_dir[2]+1]*exp_fac
            
            coeff = self.eigvecs[kind,:self.num_orbs].T #[band, orb]
            include_band = self.eigvals[kind, :].flatten()<self.efermi+self.energy_shift
            count = np.zeros(self.num_orbs)
            count[include_band] = 1
            mult_coeff = np.conj(coeff)[:,:,None]*coeff[:,None,:]*count[:,None,None] # band,orb1,orb2
            sum_coeff = np.sum(mult_coeff,axis=0)
            energyCont = scaledTB*sum_coeff[:,:,None,None,None]#np.conj(coeff)[:,full_ind[0]]*coeff[:,full_ind[1]] #[orb1,orb2,T1,T2,T3]
            cohp[kind] = energyCont * self.kpt_weights[kind]
        icohp_orb = np.sum(cohp,axis=0).real #[orb1,orb2,T1,T2,T3]
        # [s,p]*[s,p] = [s-s,s-p,p-s,p-p]
        # [s,p] * [s,p,d] = [s-s,s-p,p-s,p-p,s-d,p-d]
        icohp_atom = np.zeros((self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        ssbonds = np.zeros((self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        spbonds = np.zeros((self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        psbonds = np.zeros((self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        ppbonds = np.zeros((self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        for atm1 in range(self.numAtoms):
            for atm2 in range(self.numAtoms):
                orbs1 = np.arange(self.num_orbs)[self.orbatomnum == atm1]
                orbs2 = np.arange(self.num_orbs)[self.orbatomnum == atm2]
                sum_cohp = icohp_orb[orbs1,:][:,orbs2]
                orb_type1 = self.exactorbtype[orbs1]
                orb_type2 = self.exactorbtype[orbs2]
                # s-s
                isp1 = []
                iss1 = []
                for i in orb_type1:
                    isp1.append("p" in i)
                    iss1.append("s" in i)
                iss2 = []
                isp2 = []
                for i in orb_type2:
                    isp2.append("p" in i)
                    iss2.append("s" in i)
                sorbs1 = np.arange(len(orb_type1))[iss1]
                porbs1 = np.arange(len(orb_type1))[isp1]
                sorbs2 = np.arange(len(orb_type2))[iss2]
                porbs2 = np.arange(len(orb_type2))[isp2]
                ssbond = np.sum(sum_cohp[sorbs1,:][:,sorbs2],axis=(0,1))
                spbond = np.sum(sum_cohp[sorbs1,:][:,porbs2],axis=(0,1))
                psbond = np.sum(sum_cohp[porbs1,:][:,sorbs2],axis=(0,1))
                ppbond = np.sum(sum_cohp[porbs1,:][:,porbs2],axis=(0,1))
                icohp_atom[atm1,atm2] = np.sum(sum_cohp,axis=(0,1))
                ssbonds[atm1,atm2] = ssbond
                spbonds[atm1,atm2] = spbond
                psbonds[atm1,atm2] = psbond
                ppbonds[atm1,atm2] = ppbond
        #print(icohp_atom.shape)
        #print("cohp!",icohp_atom)
        bonds = icohp_atom.flatten()
        ssbonds = ssbonds.flatten()
        spbonds = spbonds.flatten()
        psbonds = psbonds.flatten()
        ppbonds = ppbonds.flatten()

        # get vectors to bonds
        num_bonds = len(icohp_atom.flatten())
        bond_indices = np.unravel_index(np.arange(num_bonds),(self.numAtoms,self.numAtoms,cell[0],cell[1],cell[2]))
        is_onsite = (bond_indices[2]==each_dir[0]) & (bond_indices[3]==each_dir[1]) & (bond_indices[4]==each_dir[2]) & (bond_indices[0]==bond_indices[1])
        bonds[is_onsite] = 0
        start_atom = [bond_indices[0],np.ones(num_bonds,dtype=np.int_)*each_dir[0],np.ones(num_bonds,dtype=np.int_)*each_dir[1],np.ones(num_bonds,dtype=np.int_)*each_dir[2]]
        start_ind = np.ravel_multi_index(start_atom,(self.numAtoms,cell[0],cell[1],cell[2]))
        end_atom = [bond_indices[1],bond_indices[2],bond_indices[3],bond_indices[4]]
        end_ind = np.ravel_multi_index(end_atom,(self.numAtoms,cell[0],cell[1],cell[2]))
        atom_indices = np.unravel_index(np.arange(num_total_atoms),(self.numAtoms,cell[0],cell[1],cell[2]))[0]
        atom_elems = self.elements[atom_indices]
        atom_rad = []
        for elem in atom_elems:
            atom_rad.append(all_atom_rad[elem])
        #colors = ["blue","green","orange","purple","red",
        
        # plot the bond lines
        # Create connecting lines trace
        data = []
        for b_ind in range(num_bonds):
            bond_energy = bonds[b_ind]
            if bond_energy < 0:
                style = "solid"
            else:
                style = "dash"
            if abs(bond_energy) > 0.1:
                start = start_ind[b_ind]
                end = end_ind[b_ind]
                start_atm = np.array([x_data[start],y_data[start],z_data[start]])
                end_atm = np.array([x_data[end],y_data[end],z_data[end]])
                bond_dist = np.linalg.norm(end_atm-start_atm)
                NN = np.arange(len(self.NN_dists))[np.argmin(np.abs(self.NN_dists-bond_dist))]# <= 0.0001][0]
                centerx = (x_data[start]+x_data[end])/2
                centery = (y_data[start]+y_data[end])/2
                centerz = (z_data[start]+z_data[end])/2
                data.append(go.Scatter3d(
                    x=[centerx],
                    y=[centery],
                    z=[centerz],
                    mode='markers',
                    marker=dict(size=20,color='blue',opacity=0.),
                    hoverinfo = "text",
                    text = "Bond energy: " + str(np.around(bond_energy,decimals=3))+
                                    "<br>Bond length: " + str(np.around(bond_dist,decimals=3)) + " (" + str(NN) + "NN)" +
                                    "<br>Breakdown into orbitals: <br>" +
                                    atom_elems[start] + " s - "+ atom_elems[end]+" s: " +
                                    str(np.around(ssbonds[b_ind],decimals=3)) + "<br>" +
                                    atom_elems[start] + " s - "+ atom_elems[end]+" p: " +
                                    str(np.around(spbonds[b_ind],decimals=3)) + "<br>" + 
                                    atom_elems[start] + " p - "+ atom_elems[end]+" s: " +
                                    str(np.around(psbonds[b_ind],decimals=3)) + "<br>" +
                                    atom_elems[start] + " p - "+ atom_elems[end]+" p: " +
                                    str(np.around(ppbonds[b_ind],decimals=3))
                    ))
                start_rad = atom_rad[start]
                end_rad = atom_rad[end]
                data.append(go.Scatter3d(
                    x=[start_atm[0],end_atm[0]],
                    y=[start_atm[1],end_atm[1]],
                    z=[start_atm[2],end_atm[2]],
                    mode='markers+lines',
                    marker=dict(size=[start_rad*40,end_rad*40],color=[atom_indices[start],atom_indices[end]],opacity=1.0),
                    line=dict(
                        color='black',
                        width=abs(bond_energy)*20,
                        dash = style
                    ),hoverinfo='none'))

        
        # Create figure
        fig = go.Figure(data=data)

        # update figure design
        #fig.update_traces(hovertemplate=None)
        fig.update_layout(autosize=False, width=1200, height=800)#,hovermode="x unified") 
        fig.update_layout(scene = dict(xaxis = dict(showbackground=False,showspikes=False), 
                                    yaxis = dict(showbackground=False,showspikes=False),
                                    zaxis = dict(showbackground=False,showspikes=False)))
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.show()

    def get_bandstructure(self,return_evals = False,num_kpts=100):
        #get kpoints
        #get kpath
        if not hasattr(self, 'kpoint_coords'):
            pymat_struc = struc.Structure(self._a,self.elements,self.primAtoms)
            #print(pymat_struc)
            kpts_obj = kpath.KPathSeek(pymat_struc)
            high_sym = kpts_obj.get_kpoints(1,coords_are_cartesian=False)
            
            num_each = int(num_kpts/(len(high_sym)/2))
            #print("num kpts per path:",num_each)
            kpoints = kpts_obj.get_kpoints(num_each,coords_are_cartesian=False)
            self.num_kpts = len(kpoints[0])
            #print(self.num_kpts)
            self.kpoint_coords = np.array(kpoints[0])
            self.kpoint_labels = np.array(kpoints[1])
            #high_sym_pnts = kpath_obj.kpath
            #print(self.kpoint_coords)
            #print(kpoints[1])
        
        
        eigvals = np.zeros((self.num_kpts,self.num_orbs))
        eigvecs = np.zeros((self.num_kpts,self.num_orbs,self.num_orbs),dtype=np.complex_)

        vecs = np.array(self.vec_to_trans)
        orb_pos = np.array(self.orb_redcoords)
        vec_to_orbs = vecs[None,None,:,:,:] + orb_pos[None,:,None,None,None] - orb_pos[:,None,None,None,None] #[orb1,orb2,t1,t2,t3]
        #print("shape of vec:",vec_to_orbs.shape)
        hold_vecs = np.reshape(vec_to_orbs,(self.num_orbs*self.num_orbs*self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
        cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
        dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
        self.dist_to_orbs = dist_to_orbs
        self.flat_vec_to_orbs = hold_vecs
        
        #for kpt in range(self.num_kpts):
        #        print("-",end="")
        #print("")
        for kpt in range(self.num_kpts):
            #print("-",end="")
            eigvals[kpt],eigvecs[kpt] = self.get_ham(self.kpoint_coords[kpt],return_truevec = True)
        #print("")
        self.evals = eigvals
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.kpoints = self.kpoint_coords
        self.kpt_weights = np.ones(self.num_kpts)/self.num_kpts
        if return_evals == True:
            return eigvals

    def plotBS(self, ax=None,ylim=None,color_label="",colors=None,colorhalf=None):
        """
        :param ax (matplotlib.pyplot axis): axis to save the bandstructure to, otherwise generate new axis
        :param selectedDot (1D integer array): gives kpoint and band index of the dot selected to make green circle
                                eg: [3,4]
        :return: matplotlib.pyplot axis with bandstructure plotted
        """
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


        bs_figure = go.Figure()


        #plot the bands, i loops over each band
        if color_label == "":
            for i in range(self.num_orbs):
                bs_figure.add_trace(go.Scatter(x=kpt_line,
                                   y= self.evals[:,i] - (self.efermi+self.energy_shift),
                                   marker=dict(color="black", size=2),showlegend=False))
        else: #passed from COHP and color points based on COHP
            if colorhalf == None:
                halfrange = np.average(np.abs(colors)) * 3
            else:
                halfrange = colorhalf

            for i in range(self.num_orbs):
                bs_figure.add_trace(go.Scatter(x=kpt_line,
                                   y= self.evals[:,i] - (self.efermi+self.energy_shift),mode='markers',
                                   marker=dict(color=colors[:,i],colorscale="PiYG_r",cmin=-halfrange,cmax=halfrange,colorbar=dict(title=color_label),size=12),showlegend=False))

        if ylim == None:
            minen = np.amin(self.evals.flatten() - (self.efermi+self.energy_shift))
            maxen = np.amax(self.evals.flatten() - (self.efermi+self.energy_shift))
            ylim = (minen-0.5,maxen+0.5)

        # Add custom grid lines
        for x_val in nodes:
            bs_figure.add_shape(type='line',
                                   x0=x_val, x1=x_val, y0=ylim[0], y1=ylim[1],
                                   line=dict(color='black', width=1))

        bs_figure.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bs_figure.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False,range=[kpt_line[0], kpt_line[-1]], tickvals=nodes,ticktext=labels,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bandstructure</b>', 'font':dict(size=26, color='black',family='Times')},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.5  # Center the title
        )

        return bs_figure

    def plot_hopping(self):
        #fig = plt.figure()
        actual_hoppings = copy.deepcopy(self.TB_params)

        for orb in range(self.num_orbs):
            center = ((self.num_trans-1)/2).astype(int)
            actual_hoppings[orb][orb][center[0]][center[1]][center[2]] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                #print(vec_to_orbs.shape,hold_vecs.shape,cart_hold_vecs.shape,dist_to_orbs.shape)
                dist_to_orbs = np.reshape(dist_to_orbs,(self.num_trans[0],self.num_trans[1],self.num_trans[2]))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        #print("hopping dist:",hopping_distance)
        #print("num hops:",num_hops)
        hopping_strength[hopping_strength == 0] = 10**(-6)

        #plt.get_current_fig_manager().window.raise_()
        plt.scatter(hopping_distance,np.log10(np.abs(hopping_strength)))
        plt.xlim((0,30))
        plt.show()

    def plot_overlaps(self):
        #fig = plt.figure()
        actual_hoppings = copy.deepcopy(self.overlaps_params)

        for orb in range(self.num_orbs):
            center = ((self.num_trans-1)/2).astype(int)
            actual_hoppings[orb][orb][center[0]][center[1]][center[2]] = 0
        hopping_strength = actual_hoppings.flatten().real
        hopping_distance = np.zeros((self.num_orbs,self.num_orbs,self.num_trans[0],self.num_trans[1],self.num_trans[2]))
        for orb1 in range(self.num_orbs):
            for orb2 in range(self.num_orbs):
                vec_to_orbs = self.vec_to_trans[:,:,:] + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                hold_vecs = np.reshape(vec_to_orbs,(self.num_trans[0]*self.num_trans[1]*self.num_trans[2],3)).transpose()
                cart_hold_vecs = _red_to_cart((self._a[0], self._a[1], self._a[2]), hold_vecs.transpose())
                dist_to_orbs = np.linalg.norm(cart_hold_vecs,axis=1)
                #print(vec_to_orbs.shape,hold_vecs.shape,cart_hold_vecs.shape,dist_to_orbs.shape)
                dist_to_orbs = np.reshape(dist_to_orbs,(self.num_trans[0],self.num_trans[1],self.num_trans[2]))
                hopping_distance[orb1][orb2] = dist_to_orbs
        hopping_distance = hopping_distance.flatten()
        num_hops = len(hopping_distance)
        #print("hopping dist:",hopping_distance)
        #print("num hops:",num_hops)
        hopping_strength[hopping_strength == 0] = 10**(-6)

        #plt.get_current_fig_manager().window.raise_()
        plt.scatter(hopping_distance,np.log10(np.abs(hopping_strength)))
        plt.xlim((0,30))
        plt.show()

    def get_DFT_bandstruc(self,directory):
        bandstrucKPT = Eigenval(directory+'EIGENVAL')
        band_kpts = bandstrucKPT.kpoints
        num_kpts = bandstrucKPT.nkpt
        bs_bands = bandstrucKPT.nbands
        # get DFT eigenvalues
        limited_kpts = num_kpts
        DFTeigvals = np.array(list(bandstrucKPT.eigenvalues.values()))[0,:,:,0][:limited_kpts]
        occupations = np.array(list(bandstrucKPT.eigenvalues.values()))[0,:,:,1]
        #print("kpts:",band_kpts)
        #print("evals:",DFTeigvals.shape,DFTeigvals)
        # get TB model eigenvalues
        self.num_kpts = limited_kpts#num_kpts
        self.kpoint_coords = band_kpts[:limited_kpts]
        TBeigvals = np.array(self.get_bandstructure(return_evals=True))-self.energy_shift
        # error data #in the future use occupations
        num_bands = min(bs_bands,len(TBeigvals[0]))
        num_VB = num_bands-4
        diffEig = np.abs(DFTeigvals[:,:num_bands] - TBeigvals[:,:num_bands])
        sqDiff = np.square(diffEig)
        wan_dis = (np.sum(sqDiff[:,:num_VB])/num_kpts/4)**(1/2)
        avgVBerr = np.average(diffEig[:,:num_VB])
        avgMCBerr = np.average(diffEig[:,num_VB])
        avgCBerr = np.average(diffEig[:,num_VB:])
        #print("average error in Valence Bands:", avgVBerr)
        #print("average error in Bottom CB:", avgMCBerr)
        #print("average error in Conduc Bands:", avgCBerr)
        #print("band distance in Valence Bands:", wan_dis)
        #num_bands = len(DFTeigvals[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.get_current_fig_manager().window.raise_()

        for band in range(bs_bands):
            ax.plot(range(limited_kpts), DFTeigvals[:, band] , c='black')

        for band in range(len(TBeigvals[0])):
            #for kpt in range(self.num_kpts):
            #    ax.plot(kpt, eigvals[kpt][band], 'o', c='blue', markersize=projections[kpt, band] * 6 + 0.5)
            ax.plot(range(limited_kpts), TBeigvals[:, band],'--', c='red')
        plt.show()
        
    def apply_SOC(self,soc = 0.8):
        xyz_HamSOC = np.array([[0,0,1j/2,0,0,-1/2],[0,0,0,-1j/2,1/2,0],[-1j/2,0,0,0,0,-1j/2],[0,1j/2,0,0,-1j/2,0],[0,1/2,0,1j/2,0,0],[-1/2,0,1j/2,0,0,0]])
        # 1j/2 for X1Y1, -1/2 for X1Z2, -1j/2 for X2Y2, 1/2 for X2Z1, -1j/2 for Y1X1, -1j/2 for Y1Z2, 1j/2 for Y2X2, -1j/2 for Y2Z1, 1/2 for Z1X2, 1j/2 for Z1Y2, -1/2 Z2X1, 1j/2 Z2Y1
        # double the number of orbitals to have both up and down spin orbitals
        # initialize the Ham of up-up and down-down as what is it normally with up-down as 0
        # than put the xyz_HamSOC in the onsite terms between up and down pxyz orbitals
        # than just calculate bands as usual
        regTBparams = self.TB_params
        shapes = regTBparams.shape
        Rcenter = np.array(np.floor(np.array([shapes[2],shapes[3],shapes[4]])/2),dtype=np.int_)
        #print(shapes,Rcenter)
        SoC_TBparams = np.zeros((shapes[0]*2,shapes[1]*2,shapes[2],shapes[3],shapes[4]),dtype=np.complex_)
        SoC_TBparams[:shapes[0],:shapes[1],:,:,:] = regTBparams
        SoC_TBparams[shapes[0]:,shapes[1]:,:,:,:] = regTBparams
        
        regoverlaps = self.overlaps_params
        SoC_overlaps = np.zeros((shapes[0]*2,shapes[1]*2,shapes[2],shapes[3],shapes[4]),dtype=np.complex_)
        SoC_overlaps[:shapes[0],:shapes[1],:,:,:] = regoverlaps
        SoC_overlaps[shapes[0]:,shapes[1]:,:,:,:] = regoverlaps
        #SoC_overlaps[:shapes[0],shapes[1]:,:,:,:] = regoverlaps
        #SoC_overlaps[shapes[0]:,:shapes[1],:,:,:] = regoverlaps
        
        # get p orbitals
        is_porb = []
        for orb in self.exactorbtype:
            is_porb.append('p' in orb)
        is_porb = np.array(is_porb)
        p_index = np.arange(self.num_orbs)[is_porb]
        num_porbset = int(len(p_index)/3)
        num_orbs = self.num_orbs
        for atm in range(self.numAtoms):
            elem = self.elements[atm]
            if elem == "Pb" or elem == "Bi" or elem=="Si" or elem == "As":# or elem == "Se":
                #if elem=="Si" or elem == "As" or elem == "Se":
                #    soc = soc/2
                pzpxpy = p_index[atm*3:(atm+1)*3]
                #pzpxpy = [pzpxpy[1],pzpxpy[2],pzpxpy[0]]
                # 1j/2 X1Y1, Y2X2, Z1Y2, Z2Y1
                #-1j/2 X2Y2, Y1X1, Y1Z2, Y2Z1
                # 1/2  X2Z1, Z1X2
                #-1/2  X1Z2, Z2X1
                onea = [pzpxpy[1],pzpxpy[2]+num_orbs,pzpxpy[0],pzpxpy[0]+num_orbs] #X1,Y2,Z1,Z2
                oneb = [pzpxpy[2],pzpxpy[1]+num_orbs,pzpxpy[2]+num_orbs,pzpxpy[2]] #Y1,X2,Y2,Y1
                twoa = [pzpxpy[1]+num_orbs,pzpxpy[2],pzpxpy[2],pzpxpy[2]+num_orbs] #X2,Y1,Y1,Y2
                twob = [pzpxpy[2]+num_orbs,pzpxpy[1],pzpxpy[0]+num_orbs,pzpxpy[0]] #Y2,X1,Z2,Z1
                threea = [pzpxpy[1]+num_orbs,pzpxpy[0]] # X2, Z1
                threeb = [pzpxpy[0],pzpxpy[1]+num_orbs] # Z1, X2
                foura = [pzpxpy[1],pzpxpy[0]+num_orbs] # X1, Z2
                fourb = [pzpxpy[0]+num_orbs,pzpxpy[1]] # Z2, X1
                # now set the TB params
                '''
                #soc = 0.8
                SoC_TBparams[onea,oneb,Rcenter[0],Rcenter[1],Rcenter[2]] += 1j/2*soc
                SoC_TBparams[twoa,twob,Rcenter[0],Rcenter[1],Rcenter[2]] += -1j/2*soc
                SoC_TBparams[threea,threeb,Rcenter[0],Rcenter[1],Rcenter[2]] += 1/2*soc
                SoC_TBparams[foura,fourb,Rcenter[0],Rcenter[1],Rcenter[2]] += -1/2*soc
                #print("new")
                '''
                # but now also include other terms which will be like Sij*Hsoc*Sij
                for orb1 in range(num_orbs*2):
                    for orb2 in range(num_orbs*2):
                        #print(SoC_overlaps[orb1,onea,None]*SoC_overlaps[None,oneb,orb2]*1j/2*soc)
                        tb1 = SoC_overlaps[orb1,onea,None]*SoC_overlaps[None,oneb,orb2]*1j/2*soc
                        #print(tb1.shape)
                        SoC_TBparams[orb1,orb2,:,:,:] += np.sum(SoC_overlaps[orb1,onea]*SoC_overlaps[oneb,orb2]*1j/2*soc,axis=(0))
                        SoC_TBparams[orb1,orb2,:,:,:] += np.sum(-SoC_overlaps[orb1,twoa]*SoC_overlaps[twob,orb2]*1j/2*soc,axis=(0))
                        SoC_TBparams[orb1,orb2,:,:,:] += np.sum(SoC_overlaps[orb1,threea]*SoC_overlaps[threeb,orb2]*1/2*soc,axis=(0))
                        SoC_TBparams[orb1,orb2,:,:,:] += np.sum(-SoC_overlaps[orb1,foura]*SoC_overlaps[fourb,orb2]*1/2*soc,axis=(0))
                
                        
                
        #print(regTBparams[np.arange(self.num_orbs),np.arange(self.num_orbs),Rcenter[0],Rcenter[1],Rcenter[2]])
        #print("onsite SOC:",SoC_TBparams[[0,6],:,Rcenter[0],Rcenter[1],Rcenter[2]])
        
        SOC_orbcoords = np.zeros((num_orbs*2,3))
        SOC_orbcoords[:num_orbs] = self.orb_redcoords
        SOC_orbcoords[num_orbs:] = self.orb_redcoords
        
        # now save versions with SOC
        self.num_orbs = num_orbs*2
        self.orb_redcoords = SOC_orbcoords
        self.overlaps_params = SoC_overlaps
        self.TB_params = SoC_TBparams
    
    def get_totalenergy(self,set_elec = False,num_elec = 10):
        if set_elec == False:
            # make a guess at num elec based on fermi energy
            avg_band = np.zeros(self.num_kpts)
            for kpt in range(self.num_kpts):
                num_bands = len(np.arange(self.num_orbs)[self.eigvals[kpt, :]<self.efermi+self.energy_shift])
                avg_band[kpt] = num_bands
            num_elec = int(np.round(np.average(avg_band),decimals=0))
            print("num electrons:",num_elec)
        tot_numelec = num_elec * self.num_kpts
        all_energies = np.sort(self.eigvals[:, :].flatten())
        total_energy = np.sum(all_energies[:tot_numelec])/self.num_kpts # np.sum(self.eigvals[:, :].flatten()[self.eigvals[:, :].flatten()<self.efermi+self.energy_shift])/self.num_kpts
        #print("total energy:",total_energy)
        band_gap = all_energies[tot_numelec] - all_energies[tot_numelec-1]
        gamma_bg = self.eigvals[0, num_elec] - self.eigvals[0, num_elec-1]
        #print("band gap:",band_gap)
        #print("band gap at gamma:",gamma_bg)
        
        

def reciprocal_integral(f,cell_vol):
    integral = np.sum(f)
    #scale so that is same as real space integral
    supercell_volume = cell_vol
    return integral #* supercell_volume

def complex_funs(phi,theta,parameters):
    a, b, c, d, e, f, g, h, i = parameters
    func = {}
    func["00"] = np.ones(len(phi))*(1/4/np.pi)**(1/2)
    func["10"] = np.cos(theta)*(3/4/np.pi)**(1/2)
    func["1p1"] = np.sin(theta)*np.cos(phi)*(3/8/np.pi*2)**(1/2) #* -np.exp(phi * 1j)
    func["1n1"] = np.sin(theta)*np.sin(phi)*(3/8/np.pi*2)**(1/2) # * np.exp(-phi * 1j)
    #z2,xz,yz,x2-y2,xy
    func["20"] = (3 * np.cos(theta) ** 2 - 1)*(5/16/np.pi)**(1/2)  #z^2
    func["2p1"] = np.sin(theta) * np.cos(theta)*np.cos(phi)*(15/8/np.pi*2)**(1/2) #-* np.exp(phi * 1j)  #xz
    func["2n1"] = np.sin(theta) * np.cos(theta)*np.sin(phi)*(15/8/np.pi*2)**(1/2) #* np.exp(-phi * 1j)  #yz
    func["2p2"] = np.sin(theta) ** 2 *np.cos(2*phi)*(15/32/np.pi*2)**(1/2) #* np.exp(phi * 2j)  #x^2-y^2
    func["2n2"] = np.sin(theta) ** 2 *np.sin(2*phi)*(15/32/np.pi*2)**(1/2) #* np.exp(-phi * 2j)  #xy

    return np.array([a*func["00"],b*func["10"],c*func["1p1"],d*func["1n1"],e*func["20"],f*func["2p1"],g*func['2n1'],h*func['2p2'],i*func['2n2']])

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
    (a1,a2,a3)=prim_vec
    prim = np.array(prim_coord)
    # cartesian coordinates
    cart=np.zeros_like(prim_coord,dtype=float)
    #for i in range(0,len(cart)):
    cart = [a1]*np.array([prim[:,0]]).T + [a2]*np.array([prim[:,1]]).T + [a3]*np.array([prim[:,2]]).T
    return cart

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class COHPDashApp(object):
    def __init__(self):
        made = True
        #print("making app")
    def create_cohp_dashapp(self,requests_pathname_prefix):
        import dash
        from dash import dcc, html, ALL, Dash
        from dash.dependencies import Input, Output, State

        all_directs = ["Silicon","Graphene","GaAs","PbO","PbS"]
        direct = "Silicon/"  # final_PbO/uniform/new_new/"#"LK99/hy-pb1/uniform/"##"testing_oxi/Bi2O3/prim/"#"final_Si/uniform/no_detangle/"#"final_Si/uniform/lrealFALSE/"#"final_Graphene/uniform/"##"final_Si/uniform/7x7x7/" #'final_PbO/uniform/noTimeSym/'"PbO_fmtedUNKtest/"
        self.myTB = TBModel(direct, min_hopping_dist=8)

        self.myTB.get_bandstructure(num_kpts=20)

        # Sample data
        keys = np.unique(self.myTB.elements)
        data = {}
        for elem in keys:
            elemorbs = np.array(self.myTB.exactorbtype, dtype='<U1')[np.array(self.myTB.orbatomname) == elem]
            data[elem] = list(np.unique(elemorbs)[::-1])
        #print(data)
        self.data = data
        #data = {"Si": ["s", "p"], "Pb": ["s", "p", "d"]}

        # Initialize Dash app
        import flask
        server = flask.Flask(__name__)
        app = Dash(server=server, requests_pathname_prefix=requests_pathname_prefix)
        #app = Dash(__name__)

        # Define a recursive function to create nested checklist items
        def create_nested_checklist(data, level=0,orb=0):
            checklist_items = []
            for atom, orbitals in data.items():
                label = atom if level == 0 else f"{atom} ({', '.join(orbitals)})"
                checklist_items.append(html.Div([
                                                   label,
                                                   dcc.Checklist(
                                                       id={'type': 'orbitals-checklist'+str(orb), 'index': atom},
                                                       options=[{'label':"   "+orbital, 'value': f"{atom}-{orbital}"} for
                                                                orbital in orbitals],
                                                       value=[f"{atom}-{orbital}" for orbital in orbitals],
                                                       inline=True,labelStyle= {"width":"3rem"}
                                                   ),
                                               ] + create_nested_checklist({}, level + 1)))
            return checklist_items

        # App layout
        NNs = np.append(["All"],range(1,10))
        #print(NNs)
        app.layout = html.Div([html.Div([html.H1("Compound", style={'margin': '0px', 'text-align': 'center', 'font-size': 20,'width':100}),
                          dcc.Dropdown(id='compound', options=[{'label': str(option), 'value': option} for option in all_directs],value=all_directs[0],style={'margin':'5px',"width":100})],
                         style={"width":600,'height':90}),
            html.Div([html.H1("Orbital \u03B1", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                        html.Ul(id='orbs1', children=create_nested_checklist(data,orb=0),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Orbital \u03B2", style={'margin': '5px', 'text-align': 'center', 'font-size': 22,'height':20}),
                      html.Ul(id='orbs2', children=create_nested_checklist(data,orb=1),style={'font-size':20})],style={"width":200,'display': 'inline-block'}),
                html.Div([html.H1("Nearest Neighbor", style={'margin': '5px', 'text-align': 'center', 'font-size': 20,'height':20,'width':100,'display': 'inline-block'}),
                          dcc.Dropdown(id='nearest-neighbor', options=[{'label': str(option), 'value': option} for option in NNs],value=NNs[0],style={'margin':'5px',"width":60,'display': 'inline-block'})],
                         style={"width":180,'display': 'inline-block'}),
            html.Button('Calculate new COHP', id='print-button', n_clicks=0),
            html.Div(dcc.Graph(id='COHP-figure')),
            dcc.Store(id='selected-items-store', data=[data,data])
            ],style={'height': 500,'width':600})

        # Callbacks to dynamically generate the selected items callback for each atom
        #for atom in list(self.data.keys()):
        @app.callback(
            Output('selected-items-store', 'data', allow_duplicate=True),
            Input({'type': 'orbitals-checklist0', 'index': ALL}, 'value'),
            State('selected-items-store', 'data'), allow_duplicate=True,
            prevent_initial_call=True
        )
        def update_selected_items(selected_items, stored_data):
            #print("Sel",selected_items)
            #print(dash.callback_context)
            atom = dash.callback_context.triggered_id['index']
            #print(atom)
            #elem = selected_items[0].split('-')[0]
            flat_items = [i for item in selected_items for i in item]
            new_orb = []
            for item in flat_items:
                if item.split('-')[0] == atom:
                    new_orb.append(item.split('-')[1] )
            stored_data[0][atom] = new_orb#[item.split('-')[1] for item in selected_items]
            #print(stored_data)
            return stored_data

        # Callbacks to dynamically generate the selected items callback for each atom
        #for atom in list(data.keys()):
        @app.callback(
            Output('selected-items-store', 'data', allow_duplicate=True),
            Input({'type': 'orbitals-checklist1', 'index': ALL}, 'value'),
            State('selected-items-store', 'data'),allow_duplicate=True,
            prevent_initial_call=True
        )
        def update_selected_items(selected_items, stored_data):
            #print("Sel",selected_items)
            #print(dash.callback_context)
            atom = dash.callback_context.triggered_id['index']
            #print(atom)
            #elem = selected_items[0].split('-')[0]
            flat_items = [i for item in selected_items for i in item]
            new_orb = []
            for item in flat_items:
                if item.split('-')[0] == atom:
                    new_orb.append(item.split('-')[1] )
            stored_data[1][atom] = new_orb#[item.split('-')[1] for item in flat_items]
            return stored_data

        # Callback to handle print button click and print selected items
        @app.callback(
            Output('COHP-figure', 'figure'),
            [Input('print-button', 'n_clicks')],
            [State('selected-items-store', 'data'),State('nearest-neighbor','value')],

        )
        def print_selected_items(n_clicks, stored_data,NN):
            #print("orb for COHP",stored_data)
            fig = self.myTB.get_COHP("BS",orbs=stored_data,colorhalf=15,NN=NN)
            return fig

        @app.callback(
            [Output('orbs1', 'children'),
             Output('orbs2', 'children'),
             Output('selected-items-store', 'data'),
             Output('COHP-figure', 'figure', allow_duplicate=True),
             Output('nearest-neighbor','value')],
            [Input('compound', 'value')],prevent_initial_call=True
        )

        def remake_app(direct):
            direct = direct+"/"
            self.myTB = TBModel(direct, min_hopping_dist=8)
            self.myTB.get_bandstructure(num_kpts=20)

            # Sample data
            keys = np.unique(self.myTB.elements)
            data = {}
            for elem in keys:
                elemorbs = np.array(self.myTB.exactorbtype, dtype='<U1')[np.array(self.myTB.orbatomname) == elem]
                data[elem] = list(np.unique(elemorbs)[::-1])
            #print(data)
            self.data = data

            fig = self.myTB.get_COHP("BS", orbs=[data,data], colorhalf=15)

            return create_nested_checklist(data,orb=0),create_nested_checklist(data,orb=1),[data,data],fig,"All"

        # Run the app
        #if __name__ == '__main__':

        #app.run_server(debug=True)

        return app


# my TB Silicon model
#direct =  "final_Si/uniform/lrealFALSE/" #final_PbO/uniform/new_new/"#"LK99/hy-pb1/uniform/"##"testing_oxi/Bi2O3/prim/"#"final_Si/uniform/no_detangle/"#"final_Si/uniform/lrealFALSE/"#"final_Graphene/uniform/"##"final_Si/uniform/7x7x7/" #'final_PbO/uniform/noTimeSym/'"PbO_fmtedUNKtest/"
#test = TBModel(direct,min_hopping_dist=15)

#test.get_bandstructure()
#test.plotBS()s

#BandstrucDir = "final_Si/bandstruc/" #"final_Graphene/bandstruc/"#"final_PbO/bandstruc/"#"final_Si/bandstruc/"#
#test.get_DFT_bandstruc(BandstrucDir)

#test.get_bandstructure()
#test.plot_hopping()
#test.plot_overlaps()
#test.plotBS()