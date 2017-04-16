# Provided under BSD license
# Copyright (c) 2017, Jeremy Wohlwend
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of SimExm nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JEREMY WOHLWEND BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
output.py

Handles storage of simulation outputs.
Simulation stacks can be saved in three possible formats:
    - TIFF stack
    - GIF stack
    - Image sequence (png)
In addition, one can also merge channels into rgb volumes or save each channel
separatly.

Ground truth is stored on a per-channel basis, and cells can be put in the same volume
or separated in a volume for each cell. This is useful when the expansion factor is small
and there is overlap.
"""

import os
from tifffile import imsave
from scipy.misc import imresize
import numpy as np
from PIL import Image
from images2gif import writeGif

def save_as_tiff(volume, path, name, rgb):
    """
    Saves the given volume at location path/name in TIFF format.
    
    Args:
        volume: 3D array (z, x, y)
            the volume to save
        path: string
            location to use to save the volume
        name: string
            the name of the output volume
        rgb: boolean
            whether to save the volume as an RGB stack or a single channel stack
    """
    dest = path + name + '.tiff'
    if rgb:
        imsave(dest, volume, photometric='rgb')
    else:
        imsave(dest, volume, photometric='minisblack')

def save_as_gif(volume, path, name, rgb):
    """
    Saves the given volume at location path/name in GIF format.
    
    Args:
        volume: 3D array (z, x, y)
            the volume to save
        path: string
            location to use to save the volume
        name: string
            the name of the output volume
        rgb: boolean
            whether to save the volume as an RGB stack or a single channel stack
    """
    dest = path + name + '.gif'
    sequence = [np.squeeze(volume[i]) for i in range(volume.shape[0])]
    writeGif(dest, sequence, duration=0.5)

def save_as_image_sequence(volume, path, name, rgb):
    """
    Saves the given volume at location path/name in a sequence of PNG images,
    with an image for each slice.
    
    Args:
        volume: 3D array (z, x, y)
            the volume to save
        path: string
            location to use to save the volume
        name: string
            the name of the output volume
        rgb: boolean
            whether to save the volume as an RGB stack or a single channel stack
    """
    sequence = [np.squeeze(volume[i]) for i in range(volume.shape[0])]
    digits = np.floor(np.log10(volume.shape[0]))
    dest = path + name
    if not os.path.isdir(dest):
        os.mkdir(dest)
    for i in range(volume.shape[0]):
        #Compute appropriate number of 0's to add in front of the slice number
        suffix = int(digits - np.floor(np.log10(i))) * '0' if i > 0 else int(digits) * '0'
        im_path = dest + '/image_' + suffix + str(i) + '.png'
        if rgb:
            #'RGB' saves volume as rgb images
            im = Image.fromarray(np.squeeze(volume[i]), 'RGB')
        else:
            #'L' is used to save integer images
            im = Image.fromarray(np.squeeze(volume[i]), 'L')
        im.save(im_path)

def merge(volumes):
    """
    Merges the given volumes into multiple 3 channel (RGB) volumes

    Args:
        volumes: list of numpy 3D arrays
            list of volumes to break up and stack into multiple 3-channels stacks
    Returns:
        out: list of numpy 4D arrays (z, x, y, channel)
            list of rgb volumes
    """
    out = []
    #Add missing channels to round up to % 3
    num_empty = 0 if len(volumes) % 3 == 0 else 3 - len(volumes) % 3
    for i in xrange(num_empty):
        empty = np.zeros_like(volumes[0])
        volumes.append(empty)
    #Split every 3 stacks
    for i in xrange(0, len(volumes), 3):
        vol = np.stack(volumes[i:i+3], axis = -1)
        out.append(vol.astype(np.uint8))
    return out

#The three possible saving methods
SAVE_FUNCTION = {'tiff': save_as_tiff,\
                 'gif': save_as_gif,\
                 'image sequence': save_as_image_sequence}

def save(volumes, path, name, sim_channels, format, **kwargs):
    """
    Saves the simulation stack with the given output parameters.

    Args:
        volumes: list of numpy 3D uint8 arrays
            list of volumes to store, one for each channel
        path: string
            the destination path
        name: string
            name of the simulation experiment
        sim_channels: string ('merged' or 'splitted')
            whether to store each channel in a separate stack or 
            create an RGB stack for every sequence of 3 volumes
        format: string ('tiff', 'gif' or 'image sequence')
            the desired output format
    """
    if path[-1] != "/": path += "/"
    if not os.path.isdir(path + name):
        os.mkdir(path + name)
    dest = path + name + '/simulation'
    if not os.path.isdir(dest): os.mkdir(dest)

    i = 0
    if sim_channels == 'merged':
        volumes = merge(volumes)
        for vol in volumes:
            #Get save function
            sf = SAVE_FUNCTION[format]
            #Save 3 at a time
            sf(vol, dest + '/', 'channels_{}{}{}'.format(i, i + 1, i + 2), True)
            i += 3
    else:
        for vol in volumes:
            sf = SAVE_FUNCTION[format]
            #Save each channel in a different volume
            sf(vol, dest + '/', 'channel_{}'.format(i), False)
            i += 1

def save_gt(gt_dataset, labeled_cells, volume_dim, out_dim, voxel_dim, expansion_params,
             optics_params, path, name, gt_cells, gt_region, format, **kwargs):
    """
    Saves the ground truth stack with the given output parameters.
    
    Args:
        gt_dataset: dict cell_id (string) -> region (string) -> voxels (list of (z, x, y) tuples)
            the loaded data, in simulation format, with the cell_ids as keys pointing to 
            a sub dict which points from cell regions to lists of voxels in the form 
            of (z, x, y) tuples, where each tuple is a voxel.
        labeled_cells: dict fluorophore-> list of cell_ids:
            dictionary contraining the cell_ids labeled by each of the fluorophores
        volume_dim: (z, x, y) integer tuples
            the dimensions of the original volume
        volume_dim: (z, x, y) integer tuples
            the dimensions of the output volume
        path: string
            path where to save the ground truth
        name:
            name of the experiment
        gt_cells: string, 'merged' or 'splitted'
            if merged, all the cells' ground truth are in the same volume,
            if splitted, a new volume is made for each labeled cell
        gt_region: string
            the region of the cell to put in the ground truth
        format: string 'tiff', 'png' or 'image sequence'
            the format to use to save the data, same format as the simulation output
        expansion_params: dict
            dictionary containing the expansion parameters,
            used for scaling
        optics_params: dict
            dictionary containing the optics parameters,
            used for scaling
    """
    if path[-1] != "/": path += "/"
    if not os.path.isdir(path + name):
        os.mkdir(path + name)
    dest = path + name + '/groundtruth/'
    if not os.path.isdir(dest):
        os.mkdir(dest)

    sf = SAVE_FUNCTION[format]
    for fluorophore in labeled_cells:
        if not os.path.isdir(dest + fluorophore):
            os.mkdir(dest + fluorophore)
        cells = labeled_cells[fluorophore]
        if gt_cells == 'merged':
            #Merge cells
            volume = np.zeros(volume_dim, np.uint32)
            z_step = int(np.round(volume.shape[0] / float(out_dim[0])))
            out = np.zeros(out_dim, np.uint32)
            for cell in cells:
                voxels = gt_dataset[cell][gt_region]
                #Fill volume with cell_id
                volume[tuple(voxels.transpose())] = int(cell)
                #Optical resclaing
                for i in range(0, volume.shape[0], z_step):
                    resized = imresize(volume[i], (out_dim[1], out_dim[2]), interp='nearest')
                    resized[np.nonzero(resized)] = int(cell)
                    out[i // z_step] += resized
            sf(out, dest + fluorophore + '/', 'all_cells', False)
        else:
            #Save each cell seperatly
            for cell in cells:
                #Create new volume for each cell
                volume = np.zeros(volume_dim, np.uint32)
                voxels = gt_dataset[cell][gt_region]
                volume[tuple(voxels.transpose())] = int(cell)
                #Optical rescaling
                z_step = int(np.round(volume.shape[0] / float(out_dim[0])))
                out = np.zeros(out_dim, np.uint32)
                for i in range(0, volume.shape[0], z_step):
                    out[i // z_step] = imresize(volume[i], (out_dim[1], out_dim[2]), interp='nearest')
                #This fices a bug in the interpolation which rounds the non zero value to 255
                out[np.nonzero(out)] = int(cell)
                sf(out, dest + fluorophore + '/', str(cell), False)