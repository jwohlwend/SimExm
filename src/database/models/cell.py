'''
cell.py

The Cell and CellRegion classes.

These are used to interface with the hdf5 storage format.
'''


import numpy as np


class Cell:
    '''
    Cell object representing a cell by a set of voxels in 3 dimension. 

    The different parts of the cell can be queried independently should
    they be present in the original ground truth data.

    The only added annotation is the membrane, which is computed 
    by taking edge pixels in each ground truth image.
    '''

    def __init__(self, cell_id, cell_type = "Neuron"):
      '''
      Initialize a cell object

      cell_id (int) : the id of the Cell object (taken from the ground truth to allow future reference)
      cell_type (string) : the type of the cell, by default "Neuron". Could be "glia" or more specific.
      '''
      self.cell_id = cell_id
      self.cell_type = cell_type
      self.cell_regions = {}
    
    def set_cell_type(self, cell_type):
      ''' 
      Sets the cell type of the cell 

      cell_type (string) : the type of the cell, by default "Neuron". Could be "glia" or more specific.
      '''
      self.cell_type = cell_type

    def set_full_cell(self, voxels, membrane):
      '''
      Sets the main voxels covering the cell and its membrane. This should be done before loading any region annotation.

      voxels (numpy Nx3 array, uint32) : list of (z, x, y) locations covering the cell 
      membrane (numpy Nx3 array, uint32) : list of (z, x , y) locations covering the cell's membrane
      '''
      full_region = CellRegion('full', 1, voxels, membrane)
      self.add_region(full_region)

    def get_cell_type(self):
      '''Returns the cell type  of the cell'''
      
      return self.cell_type

    def get_cell_id(self):
      '''Returns the cell_id of the cell'''
      return self.cell_id

    def get_full_cell(self, membrane_only = False):
      '''
      Returns a list of (z, x, y) locations covering the whole cell. 
      
      membrane_only (boolean) : if True, selects only membrane voxels.
      '''
      return self.get_region('full', 1, membrane_only)

    def add_region(self, region):
      '''
      Add a new region to a cell. 

      region (CellRegion) : CellRegion obect to add to the cell
      '''
      if not self.cell_regions.has_key(region.get_region_type()):
        self.cell_regions[region.get_region_type()] = {}
      self.cell_regions[region.get_region_type()][str(region.get_region_id())] = region

    def get_region_types(self):
      '''Returns all the types of region annotation present in the dataset as a list of strings.'''
      return self.cell_regions.keys()

    def get_all_region_ids(self, region_type):
      ''' 
      Returns a list of the different region_ids of the given type 

      region_type (string) : the type of the regions to query
      '''
      assert self.cell_regions.has_key(region_type), "Cell has no such region : " + region_type
      return self.cell_regions[region_type].keys()

    def get_region(self, region_type, region_id, membrane_only = False):
      '''
      Returns the given cell region as a list of voxel locations.

      region_type (int) : the type of of the region (could be synapse, dendrite, etc..)
      region_id (string) : the id of the region
      membrane_only (boolean) : if True, returns only membrane voxels
      '''
      region_id = str(region_id) #key is string for dict below
      region = self.cell_regions[region_type][region_id]
      return region.get_voxels(membrane_only)

    def get_all_regions_as_dict(self, region_type, membrane_only = False):
      ''' 
      Returns a dictionary from region_id to voxel list (numpy Nx3 array, uint32)

      region_type (string) : the type of the regions to query
      membrane_only (boolean) : if True, selects only membrane voxels
      '''
      out = {}
      for region_id in self.get_all_region_ids(region_type):
        region = self.get_region(region_type, region_id, membrane_only)
        out[str(region_id)] = region

      return out

    def get_all_regions_as_array(self, region_type, membrane_only = False):
      ''' 
      Returns a dictionary from region_id to voxel list (numpy Nx3 array, uint32)

      region_type (string) : the type of the regions to query
      membrane_only (boolean) : if True, selects only membrane voxels
      '''
      region_ids = self.get_all_region_ids(region_type)
      out = self.get_region(region_type, region_ids[0], membrane_only)
      for i in range(1, len(region_ids)):
        region = self.get_region(region_type, region_ids[i], membrane_only)
        out = np.append(out, region, axis=0)

      return out


class CellRegion:
    '''
    Class representing a region of a cell. As such, this simple class has four attributes.
    The class is a wrapper for a list of voxels decsribing the region in 3 dimensional space.
    It contains the full voxel list, a pointer to membrane voxels, as well as to its id a type.

    The class is used as a way to add new cell region annotations to a dataset
    '''

    def __init__(self, region_type, region_id, voxels, membrane):
      '''
      Initalize CellRegion object, set attributes.

      region_type (string) : the type of the region, could be synapse, dendrite, axon, soma, ...
      region_id (int) : the id of the region object (taken from the ground truth to allow future reference)
      voxels (numpy Nx3 array, uint32) : list of (x, y , z) locations covering the cell region
      membrane (numpy Nx3 array, uint32) : list of (x, y , z) locations covering the cell region's membrane
      '''
      self.voxels = voxels
      self.membrane = membrane
      self.region_id = region_id
      self.region_type = region_type

    def get_region_id(self):
      '''Returns the id of the region'''
      return self.region_id

    def get_region_type(self):
      '''Returns the type of the region'''
      return self.region_type

    def get_voxels(self, membrane_only = False):
      ''' 
      Returns all voxels covering the region 

      membrane_only (boolean) : if True, selects only membrane voxels
      '''
      out = self.membrane if membrane_only else self.voxels
      return out

