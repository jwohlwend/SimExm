'''
labeling.pyx

The BrainbowUnit class
'''

import numpy as np
cimport numpy as np


cdef class BrainbowUnit:
    '''
    BrainbowUnit

    Unit performing the labeling simulation.
    Takes a Dataset object as ground truth and generates volumes of fluorophores in space, with shape (fluorophore, X, Y, Z ).
    '''

    def __cinit__(self, gt_dataset):
        '''
        Init method, sets attributes

        gt_dataset (Dataset) : Dataset object for the labeling unit to query. See data.models.dataset
        '''
        self.gt_dataset = gt_dataset 

        self.fluo_volumes = [] # Used to stack results from multiple calls to label_cells
        self.parameters = [] # Used to stack parameters from multiple calls to label_cells
        self.labeled_cells = {} # Used to keep track of labeled cells and region for call to get_ground_truth

    cpdef label_cells(self,object region_type = 'full', object fluors = ['ATTO488'], float protein_density = 0.8, float labeling_density = 0.2,\
     int antibody_amplification_factor = 5, float fluor_noise = 0, membrane_only = True, single_neuron = False):
        '''
        Labels the cells in the volume withe the given parameter. 
        You can call label_cells multiple times and then use get_labeled_volume to retrived the labeled volume.

        region_type (string) : cell region to label. By default 'full'. To use other regions, make sure they were ingested with the ground truth
        fluors ([string]) : list of fluorphore names to use in the labeling. Each fluorophore is given a seperate channel in the output
        protein_density (float) : density of the protein stain, in number of fluorophore per nm^3. 
        labeling_density (float) : percentage of neurons to label
        antibody_amplification_factor (int) : how much to amplify the protein stain
        fluor_noise (float) : percentage of fluorophore noise to add
        membrane_only (boolean) : if True, labels only the membrane of the indicated region
        single_neuron (boolean) : if True, labels a single random neuron in the dataset
        '''
        assert (len(fluors) > 0) , "Indicate at least one fluor"
        assert (protein_density >= 0 and labeling_density >= 0 and antibody_amplification_factor >= 0),\
         "Protein density, labeling density and anitbody aamplication factor should be non negative"
        #Prepare data

        labeling_dict = {}
        cells = self.gt_dataset.get_all_cells()

        if single_neuron: #Select a single neuron
            cell_ids = cells.keys()
            cell_id = cell_ids[np.random.randint(0, len(cell_ids))]
            cells = {cell_id : cells[cell_id]} # select single neuron
            labeling_density = 1

        for cell_id in cells.keys():
            cell = cells[cell_id]
            if np.random.random_sample() > labeling_density: continue
            if not self.labeled_cells.has_key(cell_id):
                self.labeled_cells[cell_id] = {}
            self.labeled_cells[cell_id][region_type] = 1
            labeling_dict[str(cell_id)] = cell.get_all_regions_as_array(region_type, membrane_only = membrane_only)

        self.perform_labeling(labeling_dict, fluors, protein_density, antibody_amplification_factor)
        self.add_param(fluors, protein_density, labeling_density, antibody_amplification_factor, fluor_noise, region_type, membrane_only, single_neuron)

    cpdef perform_labeling(self, object labeling_dict, object fluors, float protein_density, int antibody_amplification_factor):
        '''
        Performs the labeling, using a dictionary mapping cell_ids to their voxel list. Adds a new fluorophore volume to self.fluo_volumes.
        Helper function for label_cells

        labeling_dict (dict) : dictionnary from cell_id to voxels list (numpy Nx3 array, uint32)
        fluors ([string]) : list of fluorphore names to use in the labeling. Each fluorophore is given a seperate channel in the output
        protein_density (float) : density of the protein stain, in number of fluorophore per nm^3. 
        antibody_amplification_factor (int) : how much to amplify the protein stain
        '''
        cdef int x, y, z, n, k, num_neurons, num_proteins, num_fluorophores
        cdef np.ndarray[np.uint32_t, ndim=1] neuron_list, X, Y, Z
        cdef np.ndarray[np.uint32_t, ndim=2] locations, infections
        cdef np.ndarray[np.uint32_t, ndim=4] proteins
        cdef np.ndarray[np.float64_t, ndim= 1] probabilities

        #Get neuron list
        neuron_list = np.array(labeling_dict.keys(), np.uint32)
        num_neurons = neuron_list.size
        num_fluorophores = len(fluors)

        (z, x, y) = self.gt_dataset.get_volume_dim()
        proteins = np.zeros((num_fluorophores, z, x, y), np.uint32)

        #Get infections for each neuron
        infections = self.get_infections(num_neurons, num_fluorophores)
        voxel_dim =  self.gt_dataset.get_voxel_dim()

        for n in range(num_neurons): 
            locations = np.array(labeling_dict[str(neuron_list[n])], np.uint32)
            (Z, X, Y)= np.transpose(locations)
            probabilities = np.ones(Z.size, np.float64) * 1.0/<float>Z.size

            for k in range(num_fluorophores):
                if infections[n, k] > 0:
                    num_proteins = np.random.poisson(<int>(protein_density * Z.size * np.prod(voxel_dim)))
                    protein_distribution = np.random.multinomial(num_proteins, probabilities, size=1).astype(np.uint32)
                    proteins[k, Z, X, Y] =  protein_distribution * antibody_amplification_factor

        self.fluo_volumes.append(proteins)

    
    cpdef np.ndarray[np.uint32_t,ndim=4] add_fluorophore_noise(self, np.ndarray[np.uint32_t,ndim=4] fluo_volume, int poisson_mean):
        '''Adds random number of fluorophores in the volume. In reconstruciton..'''
        cdef int x, y, z, channels
        channels, z, x, y = fluo_volume.shape[0], fluo_volume.shape[1], fluo_volume.shape[2], fluo_volume.shape[3]
        cdef np.ndarray[np.uint32_t, ndim=4] poisson_number = np.random.poisson(poisson_mean, size=(channels, z, x, y)).astype(np.uint32)
        cdef np.ndarray[np.uint32_t, ndim=4] poisson_probability = (np.random.rand(channels, z, x, y) < self.fluor_noise).astype(np.uint32)
        return np.add(fluo_volume, np.multiply(poisson_number, poisson_probability) * self.antibody_amplification_factor)

    cpdef np.ndarray[np.uint32_t,ndim=2] get_infections(self, int neuron_number, int num_fluorophores):
        '''
        Returns a 2D list indexed by neurons and giving a boolean array of infections for each neuron 

        neuron_numeber (int) : the number of neurons in the set
        num_fluorophores (int) : the number of different fluorophores that may result from a virus infection
        '''
        cdef int i, j
        cdef np.ndarray[np.uint32_t,ndim=2] infect = np.zeros((neuron_number, num_fluorophores), np.uint32)
        for i in range(neuron_number):
                while np.max(infect[i,:]) == 0:
                    for j in range(num_fluorophores):
                        infect[i,j] = np.random.random_sample() < 0.5
        return infect

    cpdef np.ndarray[np.uint32_t, ndim=4] get_labeled_volume(self):
        '''Returns the labeled volume as a (fluorophore x Z x X x Y) numpy array, uint32'''
        volumes = np.concatenate(self.fluo_volumes, 0)
        (num_volumes, z, x, y) = volumes.shape
        fluors = self.get_fluors_used()
        labeled_volume = np.zeros((len(fluors), z, x, y), np.uint32)

        for i in range(len(fluors)):
             #Find volumes of same fluorophore and merge them
            for j in range(num_volumes):
                if self.parameters[j]['fluor'] == fluors[i]:
                    labeled_volume[i, :, :, :] += volumes[j, :, :, :]

        return labeled_volume


    cpdef np.ndarray[np.uint32_t, ndim=4] get_ground_truth(self, membrane_only = False):
        ''' 
        Returns the corresponding ground truth labeling as an (X x Y x Z) numpy array, uint32 

        membrane_only (boolean) : if True, ground truth has only membranes. Tip: use False and use imageJ edge detection if you need the membranes too.
        '''
        cdef int x, y, z
        (z, x, y) = self.gt_dataset.get_volume_dim()
        cdef np.ndarray[np.uint32_t, ndim=3] ground_truth = np.zeros((z, x, y), np.uint32)
        cells = self.gt_dataset.get_all_cells()

        for cell_id in cells.keys():
            cell = cells[cell_id]
            if self.labeled_cells.has_key(cell_id):
                for region_type in self.labeled_cells[cell_id].keys():
                    locations = cell.get_all_regions_as_array(region_type, membrane_only = membrane_only)
                    (Z, X, Y) = np.transpose(locations)
                    ground_truth[Z, X, Y] = int(cell_id)

        return ground_truth

    cpdef add_param(self, object fluors, float protein_density, float labeling_density,\
      antibody_amplification_factor, float fluor_noise, object region_type, int membrane_only, int single_neuron):
        ''' 
        Adds a new set of parameters to the main parameter dictionary. 
        Each fluorophore is marked with a set of parameters.
         If label_cells is called multiple time with the same fluorophore, the parameters are appended.

        Arguments are the same as label_cells
        ''' 
        for fluor in fluors:
            #Create param dict
            params  = {}
            params['protein_density'] = protein_density
            params['labeling_density'] = labeling_density
            params['antibody_amplification_factor'] = antibody_amplification_factor
            params['fluor_noise'] = fluor_noise
            params['region_type'] = region_type
            params['membrane_only'] = membrane_only
            params['single_neuron'] = single_neuron 
            #Add to param list
            self.parameters.append({'fluor' : fluor, 'params' : params})

    cpdef object get_parameters(self):
        '''Returns a dictionary with the different parameters used throughout the labeling simulation.'''
        out = {}
        for param_dict in self.parameters:
            fluor = param_dict['fluor']
            if out.has_key(fluor):
                for param in param_dict['params'].keys():
                     out[fluor][param].append(param_dict['params'][param])
            else:
                out[fluor] = {}
                for param in param_dict['params'].keys():
                     out[fluor][param] = [param_dict['params'][param]]

        return out

    cpdef object get_fluors_used(self):
        '''Returns the total list of fluorophores used in the volume.'''
        return list(set([param['fluor'] for param in self.parameters]))

    cpdef object get_type(self):
        '''Returns the type of the labeling unit, here Brainbow.'''
        return "Brainbow"
    


