import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
opj = os.path.join

from dpu_mini.utils import *
from dpu_mini.plot_functions import *
from dpu_mini.mesh_format import *
from dpu_mini.fs_tools import *

path_to_utils = os.path.abspath(os.path.dirname(__file__))

class GenMeshMaker(FSMaker):
    '''Build on top of FSMaker (which builds on top of BaseMesh). Extend to other methods
    sub         subject name in FS dir
    fs_dir      Freesurfer files
    output_dir  Where to put files we make
    do_fs_setup Option to run without freesurfer

    Everything generic (mesh storage, colour mapping, flattening, submeshes,
    .ply export) lives in BaseMesh. This class only adds the parts that
    actually need to read from a FreeSurfer subjects directory: pulling mesh
    coordinates/curvature off disk, and label-file-based ROIs.
    '''
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        super().__init__(sub, fs_dir, output_dir=output_dir, **kwargs)
        # [1] Load mesh info
        # PULL FROM FREESURFER
        for mesh_name in ['inflated', 'sphere', 'pial', ]:
            self.mesh_info[mesh_name] = self._return_fs_mesh_coords_both_hemis(mesh=mesh_name)
        # [2] Load under surf values
        for us in ['curv', 'thickness']:
            self.us_values[us] = self._return_fs_curv_vals_both_hemis(curv_name=us, return_type='concat',)
            if us=='curv':
                vmin,vmax=-1,2
                self.us_values[us] = (self.us_values[us]<0)*1.0
            elif us=='thickness':
                vmin,vmax=0,5
            self.us_cols[us] = self._data_to_rgb(
                data=self.us_values[us],
                vmin=vmin, vmax=vmax, cmap='Greys_r',
            )

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region MESH COORDS

    def _return_fs_mesh_coords(self, mesh, hemi):
        '''Read a mesh straight off disk from the FreeSurfer subjects dir,
        then hand it to BaseMesh.add_mesh (this is the only place that
        touches FreeSurfer files directly -- everything downstream just
        works off self.mesh_info like it would for any other mesh).
        '''
        try:
            this_mesh_info = dpu_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh}'))
        except:
            this_mesh_info = dpu_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh}.T1'))

        if self.do_offsets:
            print(f'Adding offset to mesh...')

        mesh_info = self.add_mesh(
            mesh_name=mesh,
            hemi=hemi,
            coords=this_mesh_info['coords'],
            faces=this_mesh_info['faces'],
            volume_info=this_mesh_info.get('volume_info', {}),
        )
        return mesh_info

    def _return_fs_mesh_coords_both_hemis(self, mesh, return_type='dict'):
        mesh_info_D = {}
        mesh_info_L = []
        for hemi in ['lh', 'rh']:
            mesh_info_D[hemi] = self._return_fs_mesh_coords(mesh, hemi)
            mesh_info_L.append(mesh_info_D[hemi])
        if return_type=='dict':
            mesh_info = mesh_info_D
        elif return_type=='list':
            mesh_info = mesh_info_L
        return mesh_info

    def get_mesh_info(self, key):
        '''Like BaseMesh.get_mesh_info, but falls back to loading the mesh
        from the FreeSurfer subjects dir if it hasn't been loaded yet.
        '''
        if (key not in self.mesh_info.keys()) or (self.mesh_info[key] == {}):
            self.mesh_info[key] = self._return_fs_mesh_coords_both_hemis(mesh=key)
        return self.mesh_info[key]
    #endregion MESH COORDS


    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region CURV VALS 
    def _return_fs_curv_vals(self, curv_name, hemi='', **kwargs):
        '''Finds a curv file in fs folder. Returns the specified values

        curv_name       Name of curv file (can include hemi)
        hemi            hemisphere

        '''
        include = kwargs.pop('include', [])
        custom_only = kwargs.pop('custom_only', False)
        kwargs['return_msg'] = None # Don't return an error...
        kwargs['recursive'] = True
        if curv_name in ['curv', 'thickness']:
            curv_file = opj(self.sub_surf_dir,f'{hemi}.{curv_name}')
        elif custom_only: # Only look in custom folder
            curv_file = dpu_find_file_in_folder(
                filt = [curv_name, *include, hemi],
                path=self.custom_surf_dir,
                **kwargs
            )
        else:
            curv_file = dpu_find_file_in_folder(
                filt = [curv_name, *include, hemi],
                path=self.sub_surf_dir,
                **kwargs
            )
        if isinstance(curv_file, list):
            print(curv_file)
        curv_vals = dpu_read_fs_curv_file(curv_file)
        return curv_vals
    
    def _return_fs_curv_vals_both_hemis(self, curv_name, return_type='dict', **kwargs):
        curv_val_D = {}
        curv_val_L = []
        for hemi in ['lh', 'rh']:
            curv_val_D[hemi] = self._return_fs_curv_vals(curv_name, hemi, **kwargs)
            curv_val_L.append(curv_val_D[hemi])
        if return_type=='dict':
            curv_val = curv_val_D
        elif return_type=='list':
            curv_val = curv_val_L
        elif return_type=='concat':
            curv_val = np.concatenate(curv_val_L)
        return curv_val
    
    def get_us_values(self, key, ow=False):
        '''Like BaseMesh.get_us_values, but falls back to loading the curv
        file from the FreeSurfer subjects dir if it hasn't been loaded yet.
        '''
        if (key not in self.us_values.keys()) | ow:
            self.us_values[key] = self._return_fs_curv_vals_both_hemis(
                curv_name=key, return_type='concat')
        return self.us_values[key]

    #endregion CURV VALS
    
    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region ROI FUNCTIONS
    def _return_roi_bool(self, roi_name, hemi='', **kwargs):
        roi_bool = dpu_load_roi(self.sub, roi_name, fs_dir=self.fs_dir, split_LR=True)[hemi]
        return roi_bool
    def _return_roi_bool_both_hemis(self, roi_name, return_type='dict', **kwargs):
        roi_bool_D = {}
        roi_bool_L = []
        for hemi in ['lh', 'rh']:
            roi_bool_D[hemi] = self._return_roi_bool(roi_name, hemi, **kwargs)
            roi_bool_L.append(roi_bool_D[hemi])
        if return_type=='dict':
            roi_bool = roi_bool_D
        elif return_type=='list':
            roi_bool = roi_bool_L
        elif return_type=='concat':
            roi_bool = np.concatenate(roi_bool_L)
        return roi_bool
    
    # -> borders.
    def _return_roi_bool_border(self, roi_name, hemi, **kwargs):
        roi_bool = self._return_roi_bool(roi_name, hemi, **kwargs)
        roi_bool_border = dpu_find_border_vx(roi_bool, self.mesh_info['inflated'][hemi], return_type='bool')
        return roi_bool_border    

    def _return_roi_bool_border_both_hemis(self, roi_name, return_type='dict', **kwargs):
        roi_bool_D = {}
        roi_bool_L = []
        for hemi in ['lh', 'rh']:
            roi_bool_D[hemi] = self._return_roi_bool_border(roi_name, hemi, **kwargs)
            roi_bool_L.append(roi_bool_D[hemi])
        if return_type=='dict':
            roi_bool = roi_bool_D
        elif return_type=='list':
            roi_bool = roi_bool_L
        elif return_type=='concat':
            roi_bool = np.concatenate(roi_bool_L)
        return roi_bool
    
    # -> borders in order
    def _return_roi_borders_in_order(self, roi_list, **kwargs):
        '''Return the border vertices in order for each ROI in roi_list
        TODO: make generic for boolean
        '''
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        mesh_name = kwargs.get('mesh_name', 'inflated')
        combine_matches = kwargs.get('combine_matches', False)
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        roi_obj = []
        roi_count = -1
        for i_roi,roi in enumerate(roi_list):                
            roi_bool = dpu_load_roi(self.sub, roi, fs_dir=self.fs_dir, split_LR=True, combine_matches=combine_matches, recursive_search=True)
            if 'lh' not in roi_bool.keys():
                # We found extra matches!!!
                print(f'Found extra matches for {roi}')
                extra_roi_list = list(roi_bool.keys())
            else:
                extra_roi_list = [roi]
                roi_bool = {roi : roi_bool}
            for i_roi_extra, roi_extra in enumerate(extra_roi_list):
                roi_count += 1
                for ih,hemi in enumerate(hemi_list):
                    if roi_bool[roi_extra][hemi].sum()==0:
                        continue
                    border_vx_list, border_coords_list = dpu_find_border_vx_in_order(
                        roi_bool=roi_bool[roi_extra][hemi], 
                        mesh_info=self.mesh_info[mesh_name][hemi], 
                        return_coords=True,
                        )

                    for ibvx,border_vx in enumerate(border_vx_list):
                        first_instance = (ih==0) & (ibvx==0) # Only show legend for first instance
                        this_roi_dict = {
                            'hemi' : hemi,
                            'roi' : roi_extra,
                            'border_vx' : border_vx,
                            'border_coords' : border_coords_list[ibvx],
                            'first_instance' : first_instance,
                        }
                        roi_obj.append(this_roi_dict)          
        return roi_obj
    
    #endregion ROI FUNCTIONS
