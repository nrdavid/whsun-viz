
import numpy as np
from app.tb.nonorthTB import TBModel
import copy
import plotly.graph_objects as go

class Bandstructure(object):
    def __init__(self,wannierDir,wannierTag, numWannierOrbs=None,kpath = None, k_label = None):
        #super(Bandstructure, self).__init__(wannierDir, wannierTag, numWannierOrbs, kpath, k_label)
        self.wannierDir=wannierDir
        self.wannierTag=wannierTag
        self.numOrbs = int(numWannierOrbs)
        self.kptFile = wannierDir+"/"+wannierTag+'_band.kpt'
        self.datFile = wannierDir+"/"+wannierTag+'_band.dat'
        win_file = wannierDir+"/"+wannierTag+'.win'
        self.axBS = None

        #get info about structure
        filedata = open(win_file)
        filelines = filedata.readlines()
        for line in range(len(filelines)):
            if "begin unit_cell_cart" in filelines[line]:
                conv_start = line + 1
            if "begin atoms" in filelines[line]:
                atom_coord_start = line + 1
            if "end atoms" in filelines[line]:
                atom_coord_end = line
                break
        # set conventional cell vectors
        self._a1 = np.array([float(i) for i in filelines[conv_start].strip().split()])
        self._a2 = np.array([float(i) for i in filelines[conv_start + 1].strip().split()])
        self._a3 = np.array([float(i) for i in filelines[conv_start + 2].strip().split()])

        # set position and name of atoms
        self.numAtoms = int(atom_coord_end-atom_coord_start)
        self.atom_cartpos = np.zeros((self.numAtoms,3))
        self.atom_primpos = np.zeros((self.numAtoms,3))
        self.elements = np.empty((self.numAtoms),dtype=str)
        for atom in range(self.numAtoms):
            index = atom_coord_start + atom
            element = filelines[index].strip().split()[0]
            position = np.array([float(i) for i in filelines[index].strip().split()[1:4]])
            self.atom_cartpos[atom] = position
            self.atom_primpos[atom] = _cart_to_red((self._a1,self._a2,self._a3),[position])[0]
            self.elements[atom] = element
        # print("atom prim coords:",self.atom_primpos)

        #files = pythtb.w90(wannierDir,wannierTag)
        if kpath == None:
            kpath= [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.5]]
        if k_label == None:
            k_label = [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$']

        #model = files.model(min_hopping_norm=0.00001,max_distance=4,ignorable_imaginary_part=0.00001)
        (k_vec, k_dist, k_node) = self.k_path(kpath,300)# model.k_path(kpath,75)
        #(int_evals, int_evecs) = model.solve_all(k_vec, eig_vectors=True)
        #print("shape:",int_evecs.shape)

        # my TB model
        latticev = [self._a1,self._a2,self._a3]
        atoms = self.elements
        atom_coords = self.atom_primpos
        self.newModel = TBModel(wannierDir + "/", latticev, atoms, atom_coords, orbs_orth=True,min_hopping_dist=5.5)#13.62)
        file = wannierTag + "_hr.dat"# "wannier90_hr.dat"
        self.newModel.read_TBparams(file)
        self._hoppings = copy.deepcopy(self.newModel.TB_params)
        self.orb_redcoords = self.newModel.orb_redcoords

        int_evals = []
        int_evecs = []
        for kpt in k_vec:
            (evals, evecs) = self.newModel.get_ham(kpt)
            int_evals.append(evals)
            int_evecs.append(evecs)
        # end new

        self.evals = np.array(int_evals).T  #[n,k]
        self.evecs = np.array(int_evecs).transpose((2,0,1))  #[n,k,orb]
        #print("other shapes",self.evals.shape,self.evecs.shape)
        self.k_vec = k_vec
        self.k_dist = k_dist
        self.k_node = k_node
        self.kpath = kpath
        # print("knode:",k_node/k_dist[-1]*len(k_dist))
        self.k_label = k_label
        # print('intialized Bandstructure')

    def plotBS(self, ax=None, selectedDot=None, plotnew=False,ylim=None):
        """
        :param ax (matplotlib.pyplot axis): axis to save the bandstructure to, otherwise generate new axis
        :param selectedDot (1D integer array): gives kpoint and band index of the dot selected to make green circle
                                eg: [3,4]
        :return: matplotlib.pyplot axis with bandstructure plotted
        """
        fermi_energy = 4.619735#2.5529#4.619735  # best si: 7.089 #used to set the valence band maximum to zero
        #print("tick info:", self.k_node, self.k_label)

        bs_figure = go.Figure()

        if plotnew == True:
            for i in range(self.new_evals.shape[0]):
                bs_figure.add_trace(go.Scatter(x=self.k_dist,
                                               y=self.new_evals[i] - fermi_energy,
                                               marker=dict(color="red", size=2), showlegend=False))
        for i in range(self.evals.shape[0]):
            bs_figure.add_trace(go.Scatter(x=self.k_dist,
                               y=self.evals[i] - fermi_energy,
                               marker=dict(color="black", size=2),showlegend=False))
        if ylim == None:
            minen = np.amin(self.evals.flatten() - fermi_energy)
            maxen = np.amax(self.evals.flatten() - fermi_energy)
            ylim = (minen-0.5,maxen+0.5)

        # Add custom grid lines
        for x_val in self.k_node:
            bs_figure.add_shape(type='line',
                                   x0=x_val, x1=x_val, y0=ylim[0], y1=ylim[1],
                                   line=dict(color='black', width=1))

        bs_figure.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bs_figure.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False,range=[self.k_dist[0], self.k_dist[-1]], tickvals=self.k_node,ticktext=self.k_label,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bandstructure</b>', 'font':dict(size=26, color='black',family='Times')},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.5  # Center the title
        )

        #make the selected dot green
        #selectedDot = [30,4]
        if selectedDot != None:
            xcenter = self.k_dist[selectedDot[0]]
            ycenter = self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy
            # print("point center",xcenter,ycenter)
            bs_figure.add_trace(go.Scatter(x=[xcenter],y=[ycenter],
                               marker=dict(color="limegreen", size=12),showlegend=False))


        # plot new bandstructures
        #if plotnew == True:
        #    for i in range(self.evals.shape[0]):
        #        ax.plot(self.k_dist, self.new_evals[i] - fermi_energy, c='red', linewidth=2, zorder=1)

        #make the selected dot green
        #if selectedDot != None:
        #    ax.plot(self.k_dist[selectedDot[0]], self.evals[selectedDot[1]][selectedDot[0]] - fermi_energy, 'o',
        #            c='limegreen', picker=True, label=str(i), markersize=10)
        bs_figure.update_layout(autosize=False, width=800, height=900)
        return bs_figure

    def get_significant_bonds(self, band, kpoint):
        value = 0
        # define diff_evals

        #print(band, kpoint)
        diff_evals_vals = []
        diff_evals_keys = []
        orig_tb_params = []
        orbitals = np.arange(0,self.numOrbs)
        self.selected_band = band
        eigen_var = copy.deepcopy(self.evecs[band,kpoint,:])
        sig_orbitals = orbitals[abs(eigen_var)>0.1]
        #print("Sig_orbitals", sig_orbitals)
        num_trans = self.newModel.num_each_dir
        numT = int(num_trans*2+1)

        #do above the smart way that is much better
        scaledTBparams = self.newModel.get_ham(self.k_vec[kpoint],scaledTBparams=True)
        TBparams = copy.deepcopy(self._hoppings).flatten()

        energyCont = np.zeros((self.numOrbs, self.numOrbs,numT, numT, numT), dtype=np.complex_)
        for orb1 in range(self.numOrbs):
            for orb2 in range(self.numOrbs):
                energyCont[orb1,orb2] = scaledTBparams[orb1,orb2]*np.conj(eigen_var[orb1])*eigen_var[orb2]
        energyCont = energyCont.flatten()
        orb1,orb2,trans1,trans2,trans3 = np.mgrid[0:self.numOrbs,0:self.numOrbs,-num_trans:num_trans+1,-num_trans:num_trans+1,-num_trans:num_trans+1]
        (orb1, orb2, trans1, trans2, trans3) = (orb1.flatten(), orb2.flatten(), trans1.flatten(), trans2.flatten(), trans3.flatten())
        large_TB = np.abs(energyCont.real)>0.005
        energyCont = np.around(energyCont[large_TB],decimals=6)
        TBparams = TBparams[large_TB]
        (orb1, orb2, trans1, trans2, trans3) = (orb1[large_TB], orb2[large_TB], trans1[large_TB], trans2[large_TB], trans3[large_TB])
        onsite_term = [(orb1==orb2) & (trans1==0) & (trans2==0) & (trans3==0)]
        notonsite_term = np.invert(onsite_term)[0]
        energyCont = energyCont[notonsite_term]
        TBparams = TBparams[notonsite_term]
        (orb1,orb2) = (orb1[notonsite_term],orb2[notonsite_term])
        (trans1,trans2,trans3) = (trans1[notonsite_term],trans2[notonsite_term],trans3[notonsite_term])
        sorting_ind = np.flip(np.argsort(np.abs(energyCont)))
        sortedEnergyCont = energyCont[sorting_ind]
        sort_TBparams = TBparams[sorting_ind]
        (Sorb1, Sorb2, Strans1, Strans2, Strans3) = (orb1[sorting_ind], orb2[sorting_ind], trans1[sorting_ind], trans2[sorting_ind], trans3[sorting_ind])
        stSorb1 = np.char.mod('%d', Sorb1+1)
        stSorb2 = np.char.mod('%d', Sorb2+1)
        #print("to string?", Sorb1, stSorb1)
        sorted_keys = np.char.add(stSorb1,stSorb2)

        sample_keys = sorted_keys# diff_evals_keys[sorted_sample_indices]
        sorted_sample_val = sortedEnergyCont #sample_val[sorted_sample_indices] # sorted_sample_val but with +/-
        sorted_sample_mag_val = np.abs(sortedEnergyCont.real)# np.flip(np.sort(abs(sample_val))) # This used to be sorted_sample_val
        #print(sorted_sample_val)
        #print(sample_keys[:20])
        #sample_params = diff_tb_params[sorted_sample_indices]
        #print(sample_params)
        (group_mag_vals, group_vals, bool_diff, count_list, indices) = ([sorted_sample_mag_val[0]], [sorted_sample_val[0]],
                                                                        [True], [], [])
        count = 1
        groups = []
        for i in range(1, np.size(sorted_sample_mag_val)):
            #check that the parameters have the same magnitude of real energy and TB params are the same
            energy_diff = abs(abs(sorted_sample_val[i].real) - abs(sorted_sample_val[i - 1].real))
            tbval_diff = abs(abs(sort_TBparams[i])-abs(sort_TBparams[i-1]))
            same_bond_set = energy_diff < 0.004 and tbval_diff < 0.001
            #print("check params:",i,sorted_sample_val[i].real,sorted_sample_val[i - 1].real,sorted_sample_val[i].imag,sorted_sample_val[i - 1].imag,tbval_diff,same_bond_set)
            if not same_bond_set:
                group_vals.append(sorted_sample_val[i])
                group_mag_vals.append(sorted_sample_mag_val[i])
            bool_diff.append(not same_bond_set)
        #print("bool_diff", bool_diff)
        for i in range(0, len(bool_diff) - 1):
            if not bool_diff[i + 1]:
                count += 1
            else:
                count_list.append(count)
                count = 1
        count_list.append(count)

        #will recalculate groups later cause this is no longer correct
        for n in range(len(group_vals)):
            groups.append((group_vals[n]*count_list[n], count_list[n]))


        for index in range(0, np.size(sorted_sample_mag_val)):
            if sorted_sample_mag_val[index] in group_mag_vals:
                indices.append(index)
        next_ind = []
        for i in range(len(bool_diff)):
            if bool_diff[i]:
                next_ind.append(i)
        next_ind = np.array(next_ind)
        first_vec_of_keys = []
        first_vec_of_orbs = []
        first_vec_of_trans = []
        first_vec_of_tbparams_uq = []
        first_vec_of_tbparams = []
        new_groups = []
        for i in range(len(next_ind)-1):
            first_vec_of_keys.append(np.unique(sample_keys[next_ind[i]:next_ind[i+1]]))
            first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[i]:next_ind[i + 1]].real,decimals=3)).tolist())
            first_vec_of_tbparams.append(sort_TBparams[next_ind[i]:next_ind[i + 1]].tolist())
            first_vec_of_orbs.append([Sorb1[next_ind[i]:next_ind[i + 1]].tolist(),Sorb2[next_ind[i]:next_ind[i + 1]].tolist()])
            first_vec_of_trans.append([Strans1[next_ind[i]:next_ind[i + 1]].tolist(),Strans2[next_ind[i]:next_ind[i + 1]].tolist(),Strans3[next_ind[i]:next_ind[i + 1]].tolist()])
            new_groups.append((np.sum(sorted_sample_val[next_ind[i]:next_ind[i+1]]),groups[i][1]))
        # don't forget to append the last element
        first_vec_of_keys.append(np.unique(sample_keys[next_ind[-1]:]))
        first_vec_of_tbparams_uq.append(np.unique(np.around(sort_TBparams[next_ind[-1]:].real,decimals=3)).tolist())
        first_vec_of_tbparams.append(sort_TBparams[next_ind[-1]:].tolist())
        first_vec_of_orbs.append([Sorb1[next_ind[-1]:].tolist(),Sorb2[next_ind[-1]:].tolist()])
        first_vec_of_trans.append([Strans1[next_ind[-1]:].tolist(),Strans2[next_ind[-1]:].tolist(),Strans3[next_ind[-1]:].tolist()])
        new_groups.append((np.sum(sorted_sample_val[next_ind[-1]:]), groups[-1][1]))
        groups = np.array(new_groups).real
        #print("First_vec_of_keys, ",first_vec_of_keys)

        one_d_vec_of_keys = []
        for arr in first_vec_of_keys:
            con = ''
            for ind, elem in enumerate(arr):
                if ind < 6:
                    con += elem + ", "
            one_d_vec_of_keys.append(con[0:-2])
        first_vec_keys_1D = np.array(one_d_vec_of_keys)
        first_vec_tbparams_1D = np.array(first_vec_of_tbparams_uq,dtype=object)
        #print(first_vec_tbparams_1D)

        #sort based on (energy change * num params) instead of just energy
        #print(np.abs(groups[:,0]))
        #print(first_vec_keys_1D)
        new_sort = np.flip(np.argsort(np.abs(groups[:,0])))

        #print("new sorting:",new_sort)

        self.groups = np.around(groups[new_sort],decimals=3)
        self.trans = np.array(first_vec_of_trans,dtype=object)[new_sort]
        self.tbparams = np.array(first_vec_of_tbparams,dtype=object)[new_sort]
        self.orbs = np.array(first_vec_of_orbs,dtype=object)[new_sort]
        self.coeffs = eigen_var #will use this to determine phase of orbitals
        #    print("orbs",self.orbs,"  trans",self.trans)
        self.keys = first_vec_keys_1D[new_sort]
        self.tbparams_uq= first_vec_tbparams_1D[new_sort]
        self.orig_oneTB = np.zeros(6)
        #print(first_vec_tbparams_1D[new_sort].T)
        if len(new_sort) < 6:
            loop_over = range(len(new_sort))
        else:
            loop_over = range(6)
        for param in loop_over:
            uq_params =  first_vec_tbparams_1D[new_sort][param]
            one_value = np.abs(uq_params[0])
            self.orig_oneTB[param] = one_value
        #self.orig_oneTB[:len(new_sort)] = first_vec_tbparams_1D[new_sort].T[0].T
        # print("og TB", self.orig_oneTB)
        #self.new_evals = np.around(new_evals, decimals=5)
        return ([self.keys, self.groups, self.tbparams_uq])

    def plot_bond_run(self,num_bond = 0):
        # make it work for plotly.graph_objects
        #print(self.orbs)
        orbs = self.orbs[num_bond]
        trans = self.trans[num_bond]
        tbparams = self.tbparams[num_bond]
        #coeffs = self.coeffs
        #coeffs[np.abs(coeffs)>0.001] = coeffs[np.abs(coeffs)>0.001]/np.abs(coeffs)[np.abs(coeffs)>0.001]
        #phase = np.conj(coeffs[orbs[0][0]])*coeffs[orbs[1][0]]
        #print("phase from coeffs:",phase)
        ind_l = self.close_kinds[0]
        ind_r = self.close_kinds[1]
        # print(ind_l,ind_r)
        kpnts = self.k_vec[ind_l:ind_r]
        eig_vecs = self.evecs[self.selected_band, ind_l:ind_r]  # [band,kpt,orbs] --> [kpnts,orbs]

        # actually only use single point values if band is degenerate
        cur_band = self.selected_band
        ind_betwn = int((ind_l + ind_r)/2)
        if cur_band == 0 :
            same_below = False
        else:
            same_below = abs(self.evals[cur_band, ind_betwn] - self.evals[cur_band - 1, ind_betwn]) < 0.001
        if cur_band == self.numOrbs-1:
            same_above = False
        else:
            same_above = abs(self.evals[cur_band,ind_betwn]-self.evals[cur_band+1,ind_betwn])<0.001
        if same_above or same_below:
            # print("warning: bands are degenerate so only using the coefficients of selecting point for bond run")
            eig_vecs[:,:] = self.coeffs
        num_kpts = len(kpnts)

        bond_run = np.zeros((num_kpts),dtype=np.complex_)
        bond_energy = np.zeros((num_kpts),dtype=np.complex_)
        coeffs_mag = []
        for ind,k in enumerate(kpnts):
            #get and normalize eigvecs but keep phase
            coeffs = eig_vecs[ind]
            #coeffs[np.abs(coeffs) > 0.00001] = coeffs[np.abs(coeffs) > 0.00001] / np.abs(coeffs)[np.abs(coeffs) > 0.00001]
            for bond in range(len(tbparams)):
                vec = np.array([1,0,0])*trans[0][bond]+np.array([0,1,0])*trans[1][bond]+np.array([0,0,1])*trans[2][bond]
                vec = vec + self.orb_redcoords[orbs[1][bond]] - self.orb_redcoords[orbs[0][bond]]
                #print("check vec",vec)
                coeff_mag = np.abs(coeffs[orbs[0][bond]]+0.000000001)*np.abs(coeffs[orbs[1][bond]]+0.000000001)
                #print(ind,coeff_mag)
                phase_scale = np.conj(coeffs[orbs[0][bond]])*coeffs[orbs[1][bond]]/coeff_mag
                exp_fac = np.exp(-2j * np.pi * np.dot(k, vec))*phase_scale#*np.conj(coeffs[orbs[0][bond]])*coeffs[orbs[1][bond]]
                #print(np.exp(-2j * np.pi * np.dot(k, vec)),exp_fac)
                bond_run[ind] = bond_run[ind] + tbparams[bond]*exp_fac
                bond_energy[ind] = bond_energy[ind] + tbparams[bond]*exp_fac*coeff_mag
            coeffs_mag.append(coeff_mag)
        coeffs_mag = np.array(coeffs_mag)
        bond_run = bond_run #/2# seems like it's twice as large as is should be
        #print("should be zero:",bond_run.imag)
        bond_run = bond_run.real
        xaxis = np.linspace(0,1,num=num_kpts)
        #bond run should have one consistant phase

        # build go.Figure
        bondrun_fig = go.Figure()

        if not (same_above or same_below):
            bondrun_fig.add_trace(go.Scatter(x=xaxis,y=bond_run.real / 2,
                               marker=dict(color="blue", size=2),name="band-dep bond run")) # divide by 2 because maximum presence would be if orbitals were perfectly split between orb 1 and orb 2 (or 1/√2 * 1/√2)
            bondrun_fig.add_trace(go.Scatter(x=xaxis,y=bond_energy.real,
                               marker=dict(color="pink", size=2),name="bond energy"))

        else:
            bondrun_fig.add_trace(go.Scatter(x=xaxis, y=bond_run.real / 2,
                                           marker=dict(color="blue", size=2), name="band-dep bond run"))

        max = 2
        min = -2
        if (bond_run > 2).any():
            max = np.amax(bond_run / 2) + 1
        if (bond_run < -2).any():
            min = np.amin(bond_run / 2) - 1
        ylim = (min,max)

        bondrun_fig.update_layout(xaxis={'title': 'Path in k-space'},
                           yaxis={'title': 'Energy (eV)'})

        # Update layout to set axis limits and ticks
        bondrun_fig.update_layout(margin=dict(l=20, r=20, t=50, b=0),
            xaxis=dict(showgrid=False, tickvals=[0,1],ticktext=self.close_labels,
                       showline=True, linewidth=1.5, linecolor='black', mirror=True,tickfont=dict(size=16), titlefont=dict(size=18)),
            yaxis=dict(showgrid=False,range=[ylim[0], ylim[1]],showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickfont=dict(size=16), titlefont=dict(size=18)),
            title={'text':'<b>Bond run of selected bond</b>', 'font':dict(size=20, color='black',family='Times')},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_x = 0.3  # Center the title
        )
        #bondrun_fig.update_layout(autosize=False, width=600, height=350)
        return bondrun_fig

    def change_sig_bonds(self,old_vals,new_vals,tbvals=None,num_bond=None,second=False):
        """To change tight-binding parameter, need to know two orbitals and three translations"""
        # if num_bond is set than tbvals need to be a float otherwise it is a list of floats

        #reassign all TB params with the same value as the one that has been changed
        chang_model = copy.deepcopy(self.newModel)
        chang_hops = copy.deepcopy(self._hoppings)

        for i in range(len(old_vals)):
            old_val = old_vals[i]
            new_val = new_vals[i]
            #print(old_val)
            same_hop = np.around(np.abs(chang_hops),decimals=3)==np.around(np.abs(old_val),decimals=3)
            # print(chang_hops[same_hop])
            chang_hops[same_hop] = new_val*np.sign(chang_hops[same_hop])
            #print(chang_hops[same_hop])
            chang_model.TB_params = chang_hops


        int_evals = []
        int_evecs = []
        for kpt in self.k_vec:
            (evals, evecs) = chang_model.get_ham(kpt)
            int_evals.append(evals)
            int_evecs.append(evecs)

        if second == True:
            self.second_newevals = np.array(int_evals).T
        else:
            self.new_evals = np.array(int_evals).T


        '''
        # directly edit the original eigenvalues instead of recalculating
        if num_bond == None:
            if len(self.orbs) > 6:
                bonds = range(6)
            else:
                bonds = range(len(self.orbs))
            print("right length?",bonds)
        else:
            bonds = [num_bond]

        self.new_evals = copy.deepcopy(self.evals)
        for bond in bonds:
            orbs = self.orbs[bond]
            trans = self.trans[bond]
            if tbvals == None:
                # half the value
                tbparams = np.array(self.tbparams[bond]/2)
            else:
                #do 1-tbval because tbparams is substracted from evals
                tbparams = np.array(self.tbparams[bond])*(1-np.abs(np.array(tbvals[bond]/self.tbparams[bond][0])))
            for band in range(self.evals.shape[0]):
                for kind,kpt in enumerate(self.k_vec):
                    coeffs = self.evecs[band,kind]
                    for bnd in range(len(tbparams)):
                        orb1 = orbs[0][bnd]
                        orb2 = orbs[1][bnd]
                        vec = np.array([1, 0, 0]) * trans[0][bnd] + np.array([0, 1, 0]) * trans[1][bnd] + np.array([0, 0, 1]) * \
                              trans[2][bnd]
                        vec = vec + self.orb_redcoords[orb2] - self.orb_redcoords[orb1]
                        # print("check vec",vec)
                        exp_fac = np.exp(-2j * np.pi * np.dot(kpt, vec)) * np.conj(coeffs[orb1]) * coeffs[orb2]
                        self.new_evals[band,kind] = self.new_evals[band,kind] - tbparams[bnd] * exp_fac
                    if abs(self.new_evals[band,kind].imag) > 0.00001:
                        print("error! have imaginary energy",band,kind,self.new_evals[band,kind])
        self.new_evals = np.sort(self.new_evals,axis=0)
        #self.new_evals = np.array(int_evals).T  # 2D array [n,k] or [n][k] where n is band index and k is kpt index
        '''

    def k_path(self,k_list, nk):
        # taken mostly from pythTB

        # must have more k-points in the path than number of nodes
        k_list = np.array(k_list)
        if nk < k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes = k_list.shape[0]

        # extract the lattice vectors from the TB model
        lat_per = np.copy([self._a1,self._a2,self._a3])
        # choose only those that correspond to periodic directions
        # lat_per = lat_per[self._per]
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node = np.zeros(n_nodes, dtype=float)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            k_node[n] = k_node[n - 1] + dklen

        # Find indices of nodes in interpolated list
        node_index = [0]
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)

        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist = np.zeros(nk, dtype=float)
        #   array listing the interpolated k-points
        k_vec = np.zeros((nk, 3), dtype=float)

        # go over all kpoints
        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            n_i = node_index[n - 1]
            n_f = node_index[n]
            kd_i = k_node[n - 1]
            kd_f = k_node[n]
            k_i = k_list[n - 1]
            k_f = k_list[n]
            for j in range(n_i, n_f + 1):
                frac = float(j - n_i) / float(n_f - n_i)
                k_dist[j] = kd_i + frac * (kd_f - kd_i)
                k_vec[j] = k_i + frac * (k_f - k_i)

        return (k_vec,k_dist,k_node)

class Widget(Bandstructure):
    def __init__(self,wannierDir,wannierTag,numWannierOrbs,character,kpath = None, k_label = None):
        super(Widget,self).__init__(wannierDir,wannierTag,numWannierOrbs,kpath, k_label)
        self.character = character

    def plotWidget(self,requests_pathname_prefix):
        #new dash version
        import dash
        import flask
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        import dash_ag_grid as ag

        # Dash app initialization
        # if app == None:
        #     app = dash.Dash(__name__)
        server_tb = flask.Flask(__name__)
        app = dash.Dash(__name__, server=server_tb, requests_pathname_prefix=requests_pathname_prefix)

        # orbital character table info
        titles = ["orbital type", "percent character"]
        values = [[""], [0]]

        # bond run table info
        bondinfo = ['Orbitals', 'Total Energy', 'Degeneracy', 'TB parameters','Modified TB']
        bonds = [["", 0,0,'',0]]

        # Layout of the app with subplot
        app.layout = html.Div([html.H1("Bonds behind Bandstructure", style={'margin': '15px', 'text-align': 'center', 'font-size': '48px','height':80}),
            html.Div([
                html.Div(
                    dcc.Graph(
                        id='bandstructure',
                        #figure=px.scatter(data, x='sepal_width', y='sepal_length', color='species', title='Click a point')
                    ),style={'height': 800, 'width': 850, 'display': 'inline-block',"marginLeft": 20}),
                html.Div([
                    html.Div([html.H1("Orbital character of selected point", style={'margin': '0px', 'text-align': 'center', 'font-size': '20px'}),
                         ag.AgGrid(
                            id='orbital_char',
                            columnDefs=[
                                {'headerName': "titles", 'field': "titles", 'width': 170}
                            ] + [
                                {'headerName': f'Value_{i+1}', 'field': f'value_{i+1}', 'width': 100}
                                for i in range(len(values[0]))
                            ],
                            rowData=[
                                {"titles":title,**dict(zip([f'value_{i+1}' for i in range(len(values[0]))], values[index]))}
                                for index, title in enumerate(titles)
                            ],
                        dashGridOptions={"headerHeight":0},
                        style={'height': 90,'text-align': 'center'})], style={'height': 150, 'width': 700, 'display': 'inline-block'}),
                    html.Div([html.H1("Important bonds of selected point", style={'margin': '0px', 'text-align': 'center', 'font-size': '20px'}),
                        ag.AgGrid(
                                id='bond_table',
                                columnDefs=[
                                    {'headerName': header, 'field': header, 'width': 140,'editable':header=="Modified TB"} for header in bondinfo
                                ],
                                rowData=[{bondinfo[i]: bonds[row_index][i] for i in range(len(bondinfo))} for row_index in range(len(bonds))],
                                getRowStyle={"styleConditions": [{
                                    "condition": "params.rowIndex === 100",
                                    "style": {"backgroundColor": "grey"}}]},
                                style={'height': 225,'align': 'right'}
                        )], style={'height': 250, 'width': 700, 'display': 'inline-block',"marginBottom": 0}),
                    html.Div([html.Button("Calculate new", id='recalc', n_clicks=0)],style={"marginLeft": 580,"marginTop": 0,'display': 'inline-block','height':25,"marginBottom": 0}),
                    html.Div(
                        dcc.Graph(
                            id='bond_run',
                            # figure=px.scatter(data, x='sepal_width', y='sepal_length', color='species', title='Click a point')
                            ), style={'height': 400, 'width': 700, 'display': 'inline-block'})
                ], style={'height': 900, 'width': 700, 'display': 'inline-block'})
            ], style={'height': 800, 'width': 1600, 'display': 'inline-block'})
        ], style={'height': 950, 'width': 1600, 'display': 'inline-block'})

        #
        @app.callback(
            [Output('bandstructure', 'figure'),
             Output('orbital_char', 'columnDefs'),
             Output('orbital_char', 'rowData'),
             Output('bond_table', 'rowData')],
            [Input('bandstructure', 'clickData')]
        )
        def update_selected_data(clickData):
            titles = ["Orbital Type", "Percent Character"]
            values = [[""], [0]]
            bonds = [["", 0,0,'',0]]
            if clickData is None:
                # plot bandstructure
                bs_figure = self.plotBS()
                # initialize the character table
                orb_cols = [{'headerName': "titles", 'field': "titles", 'width': 170}
                        ] + [{'headerName': f'Value_{i+1}', 'field': f'value_{i+1}', 'width': 100}
                            for i in range(len(values[0]))]
                orbchar_data = [{"titles": title, **dict(zip([f'value_{i + 1}' for i in range(len(values[0]))], values[index]))}
                    for index, title in enumerate(titles)]

                # initialize the bond table
                bond_info = [{bondinfo[i]: bonds[row_index][i] for i in range(len(bondinfo))}
                 for row_index in range(len(bonds))]

            else:
                # Extracting information from clickData
                point_data = clickData['points'][0]
                # print(point_data)
                kpoint = point_data['pointIndex']
                band = point_data['curveNumber'] % self.numOrbs
                self.selected_dot = [kpoint,band]
                bs_figure = self.plotBS(selectedDot=[kpoint,band])

                # update the table
                orb_type = []
                orb_char = []
                # print(np.abs(self.evecs[band][kpoint]))
                percent_evec = np.around(np.abs(self.evecs[band][kpoint])*100,decimals=1)
                for i in range(0, self.numOrbs):  # i loops over orbitals
                    # old evecs arrangement: evecs[band][kpoint][0][i]

                    if percent_evec[i] > 5:  # only include orbitals present at point
                        # set table values
                        evec_val = percent_evec[i]
                        orb_type.append(str(i+1)+" - " +self.character[i])
                        orb_char.append(evec_val)

                values = [orb_type, orb_char]
                orb_cols = [{'headerName': "titles", 'field': "titles", 'width': 170}
                        ] + [{'headerName': f'Value_{i+1}', 'field': f'value_{i+1}', 'width': 100}
                            for i in range(len(values[0]))]
                orbchar_data = [{"titles": title, **dict(zip([f'value_{i + 1}' for i in range(len(values[0]))], values[index]))}
                    for index, title in enumerate(titles)]

                # get the bond info

                # update the second table
                keys_and_groups = self.get_significant_bonds(band=band, kpoint=kpoint)
                # self.change_sig_bonds(tbvals=0.3,num_bond=0)
                #self.ax_bond.clear()
                #self.ax_bond.set_title("Bond run")
                # self.ax_bond = self.plot_bond_run(ax=self.ax_bond,num_bond=bondind)
                num_bonds = len(keys_and_groups[2])
                #bond_vals = np.zeros((num_bonds,4))
                bond_vals = []
                org_tbparam = []
                for bond in range(num_bonds):
                    if abs(keys_and_groups[1][bond][0]) > 0.01:
                        tb_param = abs(keys_and_groups[2][bond][0])
                        [one,two,thr,fou] = [keys_and_groups[0][bond],keys_and_groups[1][bond][0],keys_and_groups[1][bond][1],str(keys_and_groups[2][bond])]
                        bond_vals.append([one,two,thr,fou,tb_param])
                        org_tbparam.append(tb_param)
                self.org_tbparam = org_tbparam
                bonds = bond_vals# [["", 1, 2, 3]]
                bond_info = [{bondinfo[i]: bonds[row_index][i] for i in range(len(bondinfo))} for row_index in range(len(bonds))]

                # get closest high sym points
                node_inds = self.k_node
                dis_ind = node_inds - self.k_dist[kpoint]
                neg_ind = copy.deepcopy(dis_ind)
                pos_ind = copy.deepcopy(dis_ind)
                neg_ind[neg_ind >= 0] = 1
                pos_ind[pos_ind < 0] = 1
                ind_right = np.argmin(pos_ind)
                ind_left = np.argmin(abs(neg_ind))
                closest_labels = [self.k_label[ind_left], self.k_label[ind_right]]
                if closest_labels[0] == closest_labels[1]:
                    ind_right = ind_right + 1
                    closest_labels[1] = self.k_label[ind_right]
                self.close_labels = closest_labels
                [dist_left, dist_right] = [self.k_node[ind_left], self.k_node[ind_right]]
                kind_l = np.abs(self.k_dist - dist_left).argmin()
                kind_r = np.abs(self.k_dist - dist_right).argmin()
                self.close_kinds = [kind_l, kind_r]

            return bs_figure, orb_cols, orbchar_data, bond_info

        @app.callback(
            [Output('bond_table', 'getRowStyle'),
            Output('bond_run', 'figure')],
            [Input('bond_table', 'cellClicked')])

        def display_selected_row(selected_rows):
            if selected_rows:
                row = selected_rows['rowIndex']
                new_style = {"styleConditions": [{
                    "condition": "params.rowIndex === " + str(row),
                    "style": {"backgroundColor": "lightgrey"}}]}
                bond_figure = self.plot_bond_run(num_bond=row)
                return new_style, bond_figure
            else:
                empty_fig = go.Figure()
                new_style = {"styleConditions": [{
                    "condition": "params.rowIndex === 100",
                    "style": {"backgroundColor": "lightgrey"}}]}
                return new_style, empty_fig

        # Callback to capture edited values and print them when the button is clicked
        @app.callback(
            Output('bandstructure', 'figure',allow_duplicate=True),
            Input('recalc', 'n_clicks'),
            State('bond_table', 'rowData'),
        prevent_initial_call=True)
        def display_edited_values(n_clicks, row_data):
            if n_clicks > 0:
                edited_values = [row['Modified TB'] for row in row_data]
                # print(edited_values)
                orig_values = self.org_tbparam
                change_old = []
                change_new = []
                for i in range(len(edited_values)):
                    if abs(orig_values[i]-edited_values[i]) > 0.001:
                        change_new.append(edited_values[i])
                        change_old.append(orig_values[i])
                # print(change_new,change_old)
                # calculate new eigenvals
                self.change_sig_bonds(old_vals=change_old,new_vals=change_new)
                bs_figure = self.plotBS(selectedDot=self.selected_dot,plotnew=True)
                return bs_figure
        return app

    def plot_hopping(self):
        self.newModel.plot_hopping()

def create_tb_app(requests_pathname_prefix):
    # emily's tb parameters

    #optional input
    characterSi = ['Si s','Si pz','Si py','Si px','Si s','Si pz','Si py','Si px',"Si s*","Si s*"]
    kpathSi = [[0.5, 0.5, 0.5],[0.0, 0.0, 0.0], [0.5, 0, 0.5], [3/4,1/4,1/2], [3/4,3/8,3/8] ,[0.0, 0.0, 0.0]]
    k_labelSi = [r'L',"\u0393",r'X',r'W',r'K',"\u0393"]

    #required input
    wanDirect = "app/tb/silicon_interact"
    wanTag = "wannier90"
    numWanSi = 8
    #initialize the widget
    model = Widget(wanDirect,wanTag,numWanSi,characterSi,kpathSi,k_labelSi)
    return model.plotWidget(requests_pathname_prefix=requests_pathname_prefix)
    
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
    prim = prim_coord
    # cartesian coordinates
    cart=np.zeros_like(prim_coord,dtype=float)
    for i in range(0,len(cart)):
        cart[i,:]=a1*prim[i][0]+a2*prim[i][1]+a3*prim[i][2]
    return cart

def normalize_wf(wavefunc,min,max,gridnum):

    integral = periodic_integral_3d(np.conj(wavefunc)*wavefunc,min,max,gridnum)
    # print("integral:",(integral)**(1/2))
    return wavefunc/(integral)**(1/2)

def periodic_integral_3d(f,a,b,n):
    func = f #3D float array
    mini = a #[0,0,0]
    maxi = b #[ax,ay,az]
    grid = n #[24,24,33]
    #max_val = np.max(func)
    #print("max val:",max_val)
    afterz_integral = np.zeros((grid[0],grid[1]))
    aftery_integral = np.zeros((grid[0],1))
    for x in range(grid[0]):
        for y in range(grid[1]):
            afterz_integral[x,y] = periodic_integral(func[x,y,:],mini[2],maxi[2],grid[2])
    for x in range(grid[0]):
        aftery_integral[x] = periodic_integral(afterz_integral[x],mini[1],maxi[1],grid[1])
    total_integral = periodic_integral(aftery_integral,mini[0],maxi[0],grid[0])
    #substract = max_val*(maxi[0]-mini[0])/grid[0]*(maxi[1]-mini[1])/grid[1]*(maxi[2]-mini[2])/grid[2]
    #print("substracted:", substract)
    return total_integral  #- substract

def periodic_integral(f, a, b, n):
    h = (b-a)/n
    integral = (f[0]+f[n-1])*0.5
    for i in range(n-1):
        integral += (f[i]+f[i+1])*0.5
    return integral*h

#character = ['Pb s','Pb d','Pb d','Pb d','Pb d','Pb d','Pb s','Pb d','Pb d','Pb d','Pb d','Pb d','O s','O pz', 'O py', 'O px','O s','O pz', 'O py', 'O px'] # for valence bands of PbO
#character = ['Pb pz','Pb py', 'Pb px','Pb pz','Pb py', 'Pb px'] # for conduciton bands of PbO
