import os
opj = os.path.join
import pickle
import sys
import numpy as np
import re
import scipy.io

import nibabel as nib
from dpu_mini.mesh_format import *
from dpu_mini.mesh_maker import GenMeshMaker
from dpu_mini.fs_tools import *
import copy as copy
def dag_auto_surf_function(**kwargs):
    '''
    ---------------------------
    Auto open a subject surface

    Args:
        p_path              path to .pkl/.npy/.gii/.mgz file 
        specific_param_path     dict with paths to specific parameters
        sub                     subject number
        fs_dir                  freesurfer director
        file_name               name of the file
        hemi_markers            How are hemispheres marked in file?
        open                    open the surface

    ''' 
    # Parse the arguments
    p_path = kwargs.pop('p_path', None)
    # MULTIPLE PATHS
    surf_list = []
    if ',' in p_path:
        p_paths = p_path.split(',')
        for i,p in enumerate(p_paths):
            tkwargs = kwargs.copy()
            tkwargs['open'] = False
            tkwargs['file_name'] = f'f{i}'
            surf_list.extend(
                dag_auto_surf_function(p_path=p, **tkwargs)
            )
        surf_list.sort()

    sub = kwargs.pop('sub', None)    
    fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])    
    roi_mask = kwargs.pop('roi_mask', 'all')    
    roi_mask = dag_load_roi(sub, roi=roi_mask, fs_dir=fs_dir, )
    if not os.path.exists(fs_dir):
        print('Could not find SUBJECTS_DIR')
        print(fs_dir)
        sys.exit()
    output_dir = kwargs.pop('output_dir', os.getcwd())
    file_name = kwargs.pop('file_name', 'auto_surf')
    hemi_markers = kwargs.pop('hemi_markers', ['lh', 'rh'])
    # Sort out how we id hemisphere
    # -> people (me) are annoyingly inconsistent with how they hame there hemispheres (I'm working on it)
    open_surf = kwargs.pop('open', True)
    pars_to_plot = kwargs.pop('pars_to_plot', None)
    if isinstance(pars_to_plot, str):
        pars_to_plot = list(pars_to_plot.split(','))
    thstr = kwargs.pop('thstr', None)
    print(kwargs)
    extra_kwargs = copy.copy(kwargs)
    
    # Check for missing stuff in param_path name
    if p_path is not None:
        if sub is None:
            sub = 'sub-'
            sub += re.search(r'sub-(.*?)_', p_path).group(1)        
    # FS OBJECT
    fs = GenMeshMaker(
        sub=sub, 
        fs_dir=fs_dir,
        )
    if surf_list != []:
        if open_surf:
            fs.open_fs_surface(surf_list, **extra_kwargs)
        else:
            return
    mask = None
    # Load some data...
    if p_path is not None:
        if '.pkl' in p_path:
            # Assume store under pars
            with open(p_path, 'rb') as f:
                pickle_dict = pickle.load(f)
            # Check for a mask 
            if 'mask' in pickle_dict.keys():
                mask = pickle_dict['mask']
            try: 
                mask = pickle_dict['config']['roi_mask']
            except:
                pass

            # Try to find pars / parameters
            if 'pars' in pickle_dict.keys():
                pars = pickle_dict['pars']
            elif 'parameters' in pickle_dict.keys():
                pars = pickle_dict['parameters']

            for k in pickle_dict.keys():
                if k in ('rsq', 'r2'):
                    pars['r2'] = pickle_dict[k].copy()
                
        elif '.npy' in p_path:
            pars = np.load(p_path)
        elif ('.gii' in p_path) or ('.mgz' in p_path):
            pars = load_mgz_or_gii(p_path, hemi_markers=hemi_markers)
    pars_dict = {}
    if not hasattr(pars, 'keys'):
        # Check if we have a 2D array (for loop below)
        if len(pars.shape)==1:
            pars = pars[:,np.newaxis]
            nvx = pars.shape[0]
        for ip in range(pars.shape[-1]):
            pars_dict[f'p{ip}'] = pars[:,ip]
    elif not isinstance(pars,dict):
        for k in pars.keys():
            pars_dict[k] = pars[k].to_numpy()
            nvx = pars_dict[k].shape[0]
    else:
        for k in pars.keys():
            pars_dict = pars[k]
            nvx = pars_dict[k].shape[0]
    vx_mask = np.ones(nvx, dtype=bool)
    if thstr is not None:
        thstrs = list(thstr.split(','))
        print(thstr)
        print(thstrs)
        for th in thstrs:
            thtype,pid,thval = th.split('-')
            if pid in pars_dict.keys():
                if thtype == 'min':
                    vx_mask &= pars_dict[pid]>float(thval)
                elif thtype == 'max':
                    vx_mask &= pars_dict[pid]<float(thval)
        print(vx_mask.mean())

    # ****************************************************
    # ****************************************************
    # FS OBJECT
    if True:
        # assume last column is rsq
        if pars_to_plot is None:
            pars_to_plot = list(pars_dict.keys())
        
        for k in pars_to_plot:
            fs.add_surface(
                data=pars_dict[k], 
                data_mask=vx_mask,
                data_sub_mask=roi_mask,
                surf_name=f'{file_name}_{k}',                
            )
                
    if open_surf:
        fs.open_fs_surface(fs.surf_list, **extra_kwargs)
    else:
        return fs.surf_list


def load_mgz_or_gii(mgz_or_gii_path, hemi_markers=['lh', 'rh']):
    '''
    Load a .mgz or .gii file and return the data (as numpy array)
    Containing both hemispheres
    '''
    mlh = [i for i in hemi_markers if 'l' in i.lower()][0]
    mrh = [i for i in hemi_markers if 'r' in i.lower()][0]

    if mlh in mgz_or_gii_path:
        lh_file = copy.copy(mgz_or_gii_path)
        rh_file = mgz_or_gii_path.replace(mlh, mrh)
    else:
        rh_file = copy.copy(mgz_or_gii_path)
        lh_file = mgz_or_gii_path.replace(mrh, mlh)
    if '.gii' in lh_file:
        lh_data = nib.load(lh_file)
        lh_data = [i.data for i in lh_data.darrays]
        lh_data = np.vstack(lh_data).squeeze()
        rh_data = nib.load(rh_file)
        rh_data = [i.data for i in rh_data.darrays]
        rh_data = np.vstack(rh_data).squeeze()
        # mgz_or_gii_data = np.concatenate([lh_data, rh_data], axis=0)
    else:
        lh_data = nib.load(lh_file).get_fdata().squeeze()[...,np.newaxis]
        rh_data = nib.load(rh_file).get_fdata().squeeze()[...,np.newaxis]
    mgz_or_gii_data = np.concatenate([lh_data, rh_data], axis=0)

    return mgz_or_gii_data