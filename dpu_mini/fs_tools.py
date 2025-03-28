import numpy as np  
import subprocess
import os
import struct
from collections import OrderedDict
opj = os.path.join

from dpu_mini.utils import *
from dpu_mini.plot_functions import *


path_to_utils = os.path.abspath(os.path.dirname(__file__))

class FSMaker(object):
    '''Used to make a freesurfer file, and view a surface in freesurfer. 
    Will create a curv file in subjects freesurfer dir, and load it a specific colormap 
    saved as the relevant command
    '''
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], **kwargs):
        
        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        print(f'Using fs dir = {self.fs_dir}')
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.sub_label_dir = opj(fs_dir, sub, 'label')
        #
        self.custom_surf_dir = opj(self.sub_surf_dir, 'custom')         # Where to put the surfaces we make
        n_vx, n_faces = dag_load_nfaces_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]}
        self.total_n_vx = sum(n_vx)
        self.n_faces = {'lh':n_faces[0], 'rh':n_faces[1]}
        self.total_n_faces = sum(n_faces)
        self.overlay_str = {}
        self.open_surf_cmds = {}
        self.surf_list = []

    def add_surface(self, data, surf_name, **kwargs):
        '''
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file

        '''
        if not os.path.exists(self.custom_surf_dir):
            print('making a custom dir')
            os.mkdir(self.custom_surf_dir)        

        exclude_as_nan = kwargs.get("exclude_as_nan", False) # Set masked values to NAN
        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        # Load colormap properties: (cmap, vmin, vmax)
        # vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
        # Get the overlay custom str and overlay to save...
        overlay_custom_str, overlay_to_save = dag_make_overlay_str(masked_data=data[data_mask], **kwargs)
        
        data_masked = np.zeros_like(data, dtype=float)
        data_masked[data_mask] = data[data_mask]
        exclude_min_val = vmin - 1
        data_masked[~data_mask] = exclude_min_val
        if exclude_as_nan:
            data_masked[~data_mask] = np.nan


        # SAVE masked data AS A CURVE FILE
        lh_masked_param = data_masked[:self.n_vx['lh']]
        rh_masked_param = data_masked[self.n_vx['lh']:]

        # now save results as a curve file
        print(f'Saving {surf_name} in {self.custom_surf_dir}')

        n_faces = dag_load_nfaces(self.sub, self.fs_dir)
        dag_write_curv(
            fn=opj(self.custom_surf_dir, f'lh.{surf_name}'), 
            curv=lh_masked_param, 
            fnum=n_faces[0])
        dag_write_curv(
            fn=opj(self.custom_surf_dir, f'rh.{surf_name}'), 
            curv=rh_masked_param, 
            fnum=n_faces[1])        
        # write_morph_data(opj(self.custom_surf_dir, f'lh.{surf_name}'),lh_masked_param)
        # write_morph_data(opj(self.custom_surf_dir, f'rh.{surf_name}'),rh_masked_param)        
        
        dag_str2file(filename=opj(self.custom_surf_dir, f'{surf_name}_overlay'),txt=overlay_to_save)
        self.overlay_str[surf_name] = overlay_custom_str        

        # Check, if it is already in the surf list:
        # if it is remove it and add it at the end
        # if not add it at the end
        if surf_name in self.surf_list:
            self.surf_list.remove(surf_name)
        self.surf_list.append(surf_name)


    def open_fs_surface(self, surf_name=[], **kwargs):
        # surf name - which surface to load...        
        fs_cmd = self.write_fs_cmd(surf_name=surf_name, **kwargs)
        print(fs_cmd)
        # self.save_fs_cmd(surf_name, **kwargs)        
        # os.chdir(self.sub_surf_dir) # move to freeview dir        
        # os.system(fs_cmd)     
        subprocess.run(fs_cmd, shell=True, cwd=self.sub_surf_dir)   

    def open_fs_surface_FIND(self, include=[], exclude=[], **kwargs):
        surf_name = dag_find_file_in_folder(
            filt=include,
            path=self.surf_list,
            exclude=exclude,
        )
        print(surf_name)
        # surf name - which surface to load...        
        fs_cmd = self.write_fs_cmd(surf_name=surf_name, **kwargs)
        # self.save_fs_cmd(surf_name, **kwargs)        
        # os.chdir(self.sub_surf_dir) # move to freeview dir        
        # os.system(fs_cmd)
        subprocess.run(fs_cmd, shell=True, cwd=self.sub_surf_dir)

    def save_fs_cmd(self, surf_name=[], **kwargs):
        cmd_name = kwargs.get('cmd_name', f'{surf_name}_cmd.txt')
        print(f'Custom overlay string saved here: ({opj(self.custom_surf_dir, cmd_name)})')
        fs_cmd = self.write_fs_cmd(surf_name=surf_name, **kwargs)
        dag_str2file(filename=opj(self.custom_surf_dir, cmd_name),txt=fs_cmd)
        
    def write_fs_cmd(self, surf_name=[], **kwargs):
        '''
        Write the bash command to open the specific surface with the overlay

        **kwargs 
        surf_name       which surface(s) to open (of the custom ones you have made)
        mesh_list       which mesh(es) to plot the surface info on (e.g., inflated, pial...)
        hemi_list       which hemispheres to load
        roi_list        which roi outlines to load
        roi_list_excl   any roi to exclude...
        roi_col_spec    if loading rois, what color? If not specified will do different colors for each nes     
        roi_mask        mask by roi?
        keep_running    keep running the command (use "&" at the end of the command). Useful if you want to take many screen shots.
        shading_off     Turn of shading? i.e., don't make it darker underneath. Default is false        
        do_scr_shot     bool            take a screenshot of the surface when it is loaded?
        scr_shot_file   str             Where to put the screenshot. If not specified goes in custom surface dir
        ss_mag          int             Magnification factor for screenshot (i.e., quality)
        ss_trim         bool            Autotrim the image 
        azimuth         float           camera angle(0-360) Default: 0
        zoom            float           camera zoom         Default: 1.00
        elevation       float           camera angle(0-360) Default: 0
        roll            float           camera angle(0-360) Default: 0        
        do_col_bar      bool            show color bar at the end. Default is true
        extra_args      str             Extra arguments added at the end... 
        f_extra_args    str             Extra arguments added at after loading surface... 
        '''
        mesh_list = kwargs.get('mesh_list', ['inflated'])
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        roi_list = kwargs.get('roi_list',None)
        roi_list_excl = kwargs.get('roi_list_excl',[])
        roi_col_spec = kwargs.get('roi_col_spec', None)
        roi_mask = kwargs.get('roi_mask', None)
        keep_running = kwargs.get('keep_running', False) # open window and keep running
        shading_off = kwargs.get('shading_off', False) # Turn shading off
        shading_off_str = ''
        if shading_off:
            shading_off_str = ':no_shading=1'

        do_scr_shot     = kwargs.get('do_scr_shot', False)
        scr_shot_file   = kwargs.get('scr_shot_file', None)
        ss_mag          = kwargs.get('ss_mag', 5)
        ss_trim         = kwargs.get('ss_trim', True)
        # *** CAMERA ANGLE ***
        cam_azimuth     = kwargs.get('azimuth', 90)
        cam_zoom        = kwargs.get('zoom', 1)
        cam_elevation   = kwargs.get('elevation', 0)
        cam_roll        = kwargs.get('roll', 0)
        # *** COLOR BAR ***
        do_col_bar  = kwargs.get('do_col_bar', True)
        # extra args
        extra_args = kwargs.get('extra_args', '')       # Extra commands at the end
        f_extra_args = kwargs.get('f_extra_args', '')   # Extra commands after a surface (-f)


        do_surf = True
        if isinstance(surf_name, str):
            surf_name=[surf_name]                 
        if len(surf_name)==0:
            do_surf = False

        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        if not isinstance(surf_name, list):
            surf_name = [surf_name]        
        
        # Prepare for roi stuff
        do_rois = False
        if roi_list is not None:
            do_rois = True
            sorted_roi_list = self.get_lr_roi_list(roi_list, roi_list_excl)

        if do_scr_shot:     
            if scr_shot_file is None:
                # Not specified -save in custom surf dir
                this_sname = 'ss' if len(surf_name)==0 else surf_name[0]
                scr_shot_file = opj(self.custom_surf_dir, f'{this_sname}_az{cam_azimuth}_z{cam_zoom}_e{cam_elevation}_r{cam_roll}.png')
            if os.path.isdir(scr_shot_file):
                # Folder specified, add the name...
                full_ss_file = opj(scr_shot_file, surf_name[0])
            else:
                # Specific file specified
                full_ss_file = scr_shot_file
            # MAKE SURE IT ENDS WITH .png
            if not full_ss_file.endswith('.png'):
                full_ss_file += '.png'
            # Add the file + the magnification and the autotrim
            auto_trim = 'autotrim ' if ss_trim else ' '
            scr_shot_flag = f"--ss {full_ss_file} {ss_mag} {auto_trim}" #{int(ss_trim)} "
            # print(scr_shot_flag)
        else:
            scr_shot_flag = ""


        if do_col_bar:
            col_bar_flag = '--colorscale'
        else:
            col_bar_flag = ''

        fs_cmd = f'freeview -f '
        for mesh in mesh_list:
            for this_hemi in hemi_list:
                fs_cmd += f' {this_hemi}.{mesh}'
                fs_cmd += f_extra_args
                if do_rois:
                    for i_roi, roi in enumerate(sorted_roi_list[this_hemi]):
                        if roi_col_spec is None:
                            roi_col = dag_get_col_vals(i_roi, 'jet', 0, len(roi_list))
                            roi_col = f'{int(roi_col[0]*255)},{int(roi_col[1]*255)},{int(roi_col[2]*255)}'
                        else:
                            roi_col = roi_col_spec
                        # this_roi_path = self.get_roi_file(roi, this_hemi)                        
                        fs_cmd += f':label={roi}:label_outline=True:label_visible=True:label_color={roi_col}' # false...
                if do_surf:
                    for this_surf_name in surf_name:
                        # this_surf_path = opj(self.custom_surf_dir, f'{this_hemi}.{this_surf_name}')                
                        this_surf_path = self.get_surf_path(this_hemi=this_hemi, this_surf_name=this_surf_name)
                        this_overlay_str = self.get_overlay_str(this_surf_name, **kwargs)
                        fs_cmd += f':overlay={this_surf_path}:{this_overlay_str}'                        
                        fs_cmd += shading_off_str
                        if roi_mask is not None:
                            this_roi_path = self.get_roi_file(roi, this_hemi)
                            fs_cmd += f':overlay_mask={this_roi_path}'       
                else:
                    fs_cmd += f':curvature_method=binary'
                    fs_cmd += shading_off_str
        fs_cmd +=  f' --camera Azimuth {cam_azimuth} Zoom {cam_zoom} Elevation {cam_elevation} Roll {cam_roll} '
        fs_cmd += f'{col_bar_flag} {scr_shot_flag}'
        fs_cmd += ' --verbose  --viewport 3d --viewsize 99999 99999'        
        fs_cmd += f' --nocursor '
        fs_cmd += extra_args
        if keep_running:
            fs_cmd += ' &'
        print(fs_cmd)
        return fs_cmd 

    def get_lr_roi_list(self, roi_list, roi_list_excl):
        '''
        Sort out the list of rois... per hemi
        Include make it capable of dealing with missing rois
        And fining matching ones
        '''
        sorted_roi_list = {
            'lh':[],
            'rh':[],
        }
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        for roi_name in roi_list:
            for hemi in ['lh', 'rh']:
                this_roi_path = dag_find_file_in_folder(
                    filt=[roi_name, hemi],
                    path=self.sub_label_dir,
                    recursive=True,
                    exclude=['._', '.thresh'] + list(roi_list_excl),
                    return_msg=None,
                    )
                if this_roi_path is not None:
                    if isinstance(this_roi_path, list):
                        sorted_roi_list[hemi] += this_roi_path
                    else:
                        sorted_roi_list[hemi].append(this_roi_path)

        return sorted_roi_list

    def get_roi_file(self, roi_name, hemi):
        roi = dag_find_file_in_folder(
            filt=[roi_name, hemi],
            path=self.sub_label_dir,
            recursive=True,
            exclude=['._', '.thresh'],
            return_msg=None,
            )
        if isinstance(roi, list):
            roi = roi[0]
        roi_path = opj(self.sub_label_dir, roi)
        return roi_path
    def get_surf_path(self, this_hemi, this_surf_name):
        # [1] Check if it exists in the custom surf dir
        this_surf_path = opj(self.custom_surf_dir, f'{this_hemi}.{this_surf_name}')
        if os.path.exists(this_surf_path):
            pass
        else: 
            # Now we need to look a bit deeper
            this_surf_path = dag_find_file_in_folder(
                filt=[this_hemi, f'.{this_surf_name}'],
                exclude=['pial'],
                path=self.sub_surf_dir,
                recursive=True,
                return_msg=None,
            )

        return this_surf_path
    
    def get_overlay_str(self, surf_name, overlay_cmap=None, **kwargs):
        overlay_str_ow = kwargs.get('overlay_str', None)
        if overlay_str_ow is not None:
            if isinstance(overlay_str_ow, list):
                # join together as a string
                overlay_str_ow = ','.join(overlay_str_ow)
            # :colormap=grayscale
            return overlay_str_ow
        
        if overlay_cmap is not None:
            overlay_str, _ = dag_make_overlay_str(cmap=overlay_cmap, **kwargs)
            return overlay_str
        if surf_name in self.overlay_str.keys():
            overlay_str = self.overlay_str[surf_name]
            return overlay_str

        # Not found in struct: check the custom surf dir...
        overlay_str = ':overlay_custom='            
        print(f'{surf_name} not in dict')
        print(f'Checking custom surf dir')
        overlay_str_file = dag_find_file_in_folder(
            filt=[surf_name, 'overlay'],
            path=self.sub_surf_dir,
            recursive=True,
            return_msg=None,
        )

        if overlay_str_file is None:
            overlay_str = ''#'greyscale :colormap=grayscale' #  grayscale/lut/heat/jet/gecolor/nih/pet/binary
        elif isinstance(overlay_str_file, list):
            overlay_str += overlay_str_file[0]
        else:
            overlay_str += overlay_str_file

        return overlay_str

    def clean_custom_surf_dir(self, do_all=False, include=[], exclude=[], sure=False):
        '''
        Remove files in the custom surf dir
        '''
        # REMOVE ALL    
        if do_all:            
            print('Removing all files')
            if not sure:
                print('Are you sure? (y/n)')
                if input() != 'y':
                    print('Exiting...')
                    return
            for file in os.listdir(self.custom_surf_dir):
                os.remove(opj(self.custom_surf_dir, file))
            return
        # REMOVE SPECIFIC
        if (include==[]) & (exclude==[]):
            print('Include or exclude must be specified')
            print('Removing nothing')
            return

        surf_list = dag_find_file_in_folder(
            filt = include,
            path=self.custom_surf_dir,
            exclude = exclude, 
            recursive=True,
            return_msg=None
        )        
        if isinstance(surf_list, str):
            surf_list = [surf_list]
        
        if surf_list is None: 
            print('No surfaces found')
            return
        
        if not sure:
            print(f'Are you sure you want to remove the following? (y/n)')
            print(surf_list)
            if input() != 'y':
                print('Exiting...')
                return

        for surf in surf_list:
            os.remove(surf)
        
        return

# ************************************************************************
# ************************************************************************
# *************************** SUPPORTING FUNCTIONS ***********************
def dag_load_nverts(sub, fs_dir = os.environ['SUBJECTS_DIR']):    
    '''
    nverts (points) in a given mesh
    '''
    n_verts, n_faces = dag_load_nfaces_nverts(sub, fs_dir)
    return n_verts

def dag_load_nfaces(sub, fs_dir=os.environ['SUBJECTS_DIR']):
    '''
    nfaces (triangular) in a given mesh
    '''
    n_verts, n_faces = dag_load_nfaces_nverts(sub, fs_dir)
    return n_faces

def dag_load_nfaces_nverts(sub, fs_dir=os.environ['SUBJECTS_DIR']):
    """
    Adapted from pycortex https://github.com/gallantlab/pycortex
    Load the number of vertices and faces in a given mesh
    """    
    n_faces = []
    n_verts = []
    for i in ['lh', 'rh']:
        surf = opj(fs_dir, sub, 'surf', f'{i}.inflated')
        with open(surf, 'rb') as fp:
            #skip magic
            fp.seek(3)
            fp.readline()
            comment = fp.readline()            
            i_verts, i_faces = struct.unpack('>2I', fp.read(8))
            n_verts.append(i_verts)    
            n_faces.append(i_faces)    
    return n_verts, n_faces


def dag_load_roi(sub, roi, fs_dir=os.environ['SUBJECTS_DIR'], split_LR=False, do_bool=True, **kwargs):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - hemi specific idx...
    '''
    need_both_hemis = kwargs.get('need_both_hemis', False) # Need ROI in both hemispheres to return true
    combine_matches = kwargs.get('combine_matches', False) # If multiple matches combine them...    
    # Get number of vx in each hemi, and total overall...
    n_verts = dag_load_nverts(sub=sub, fs_dir=fs_dir)
    total_num_vx = np.sum(n_verts)
    
    # ****************************************
    # SPECIAL CASES [all, occ, demo]
    if 'all' in roi :        
        if split_LR:
            roi_idx = {}
            roi_idx['lh'] = np.ones(n_verts[0], dtype=bool)
            roi_idx['rh'] = np.ones(n_verts[1], dtype=bool)
        else:
            roi_idx = np.ones(total_num_vx, dtype=bool)
        return roi_idx    
    elif 'demo' in roi:
        if '-' in roi:
            n_demo = int(roi.split('-')[-1])
        else:
            n_demo = 100
        if split_LR:
            roi_idx = {}
            roi_idx['lh'] = np.zeros(n_verts[0], dtype=bool)
            roi_idx['rh'] = np.zeros(n_verts[1], dtype=bool)
            roi_idx['lh'][:n_demo] = True
            roi_idx['rh'][:n_demo] = True

        else:
            roi_idx = np.zeros(total_num_vx, dtype=bool)        
            roi_idx[:n_demo] = True

        return roi_idx
    
    elif '+' in roi:
        roi = roi.split('+')
    # ****************************************
        
    # Else look for rois in subs freesurfer label folder
    roi_dir = opj(fs_dir, sub, 'label')    
    if not isinstance(roi, list): # roi can be a list 
        roi = [roi]    

    roi_idx = []
    roi_idx_split = {'lh':[], 'rh':[]}
    for this_roi in roi:    
        # Find the corresponding files
        if 'not' in this_roi:
            do_not = True
            this_roi = this_roi.split('-')[-1]
        else:
            do_not = False
        roi_file = {}
        missing_hemi = False # Do we have an ROI for both hemis? 
        for hemi in ['lh', 'rh']:
            roi_file[hemi] = dag_find_file_in_folder([this_roi, '.thresh', '.label', hemi], roi_dir, recursive=True, return_msg=None)
            # Didn't find it? Try again without "thresh"
            if roi_file[hemi] is None:
                roi_file[hemi] = dag_find_file_in_folder([this_roi, '.label', hemi], roi_dir,exclude='._', recursive=True, return_msg = None)                
            # Did we find it now? 
            if roi_file[hemi] is None:
                # If not make a note - no entry for this hemi
                missing_hemi = True
            else:        
                if (isinstance(roi_file[hemi], list)) & (not combine_matches):
                    # If we want an exact match (1 file only) 
                    # BUT we find multiple files, raise an error                    
                    print(f'Multiple matches for {this_roi} in {roi_dir}')
                    print([i.split('/')[-1] for i in roi_file[hemi]])
                    raise ValueError
                                
                elif isinstance(roi_file[hemi], list):
                    # Print which files we will be combining
                    # print('Combining')
                    # print([i.split('/')[-1] for i in roi_file[hemi]])
                    pass
                else:
                    # 1 matched file - convert to list...
                    # -> so we can loop through later
                    roi_file[hemi] = [roi_file[hemi]]

        # CHECK IF WE NEED BOTH HEMIS AND HAVE BOTH HEMIS!!
        if need_both_hemis and missing_hemi:
            print(f'Missing ROI in one hemisphere')
            print(roi_file)
            raise ValueError

        # START LOOP TO GET BOOLEAN FOR THE ROI
        LR_bool = []
        for i,hemi in enumerate(['lh', 'rh']):
            if roi_file[hemi] is None:
                idx_int = []
            else:
                # Loop through the files to combine together...
                # all the (numbered indexes of the roi files)
                idx_int = []
                for this_roi_file in roi_file[hemi]:
                    with open(this_roi_file) as f:
                        contents = f.readlines()            
                    this_idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
                    this_idx_int = [int(i) for i in this_idx_str]
                    idx_int += this_idx_int
                # Remove not unique values 
                idx_int = list(set(idx_int))
            # Option to make boolean array
            if do_bool:
                this_bool = np.zeros(n_verts[i], dtype=int)
                this_bool[idx_int] = True

            if do_not:            
                this_bool = ~this_bool

            LR_bool.append(this_bool)
        this_roi_mask = np.concatenate(LR_bool)
        roi_idx.append(this_roi_mask)
        roi_idx_split['lh'].append(LR_bool[0]) 
        roi_idx_split['rh'].append(LR_bool[1])

    roi_idx = np.vstack(roi_idx)
    roi_idx_split['lh'] = np.vstack(roi_idx_split['lh'])
    roi_idx_split['rh'] = np.vstack(roi_idx_split['rh'])
    if do_bool:
        roi_idx = roi_idx.any(0)
        roi_idx_split['lh'] = roi_idx_split['lh'].any(0)
        roi_idx_split['rh'] = roi_idx_split['rh'].any(0)    
    else:
        roi_idx = np.squeeze(roi_idx)
        roi_idx_split['lh'] = np.squeeze(roi_idx_split['lh'])
        roi_idx_split['rh'] = np.squeeze(roi_idx_split['rh'])

    if split_LR:
        return roi_idx_split
    else:
        return roi_idx

def dag_roi_list_expand(sub, roi_list, fs_dir=os.environ['SUBJECTS_DIR'] ):
    if not isinstance(roi_list, list):
        roi_list = [roi_list]
    roi_dir = opj(fs_dir, sub, 'label')    
    roi_list_expanded = []
    for roi in roi_list:                
        roi_files = dag_find_file_in_folder([roi, '.label'], roi_dir,exclude='._', recursive=True, return_msg = None)                        
        for this_roi_file in roi_files:
            this_roi_file = this_roi_file.split('/')[-1]
            this_roi_file = this_roi_file.replace('.label', '')
            this_roi_file = this_roi_file.replace('lh.', '')
            this_roi_file = this_roi_file.replace('rh.', '')
            roi_list_expanded.append(this_roi_file)
    # remove duplicates
    roi_list_expanded = list(set(roi_list_expanded))

    # Now check if we have any which match each other 
    # (i.e., if we have a "V1" and a "V1d", we should disambiguate V1 by making it V1.)
    roi_list_expanded.sort()
    for i,roi in enumerate(roi_list_expanded):
        for j,roi2 in enumerate(roi_list_expanded):
            if i==j:
                continue
            if roi2.startswith(roi):
                roi_list_expanded[i] = roi + '.'
    return roi_list_expanded

def dag_write_curv(fn, curv, fnum):
    ''' Adapted from https://github.com/simnibs/simnibs
    
    Writes a freesurfer .curv file

    Parameters
    ------------
    fn: str
        File name to be written
    curv: ndaray
        Data array to be written
    fnum: int
        Number of faces in the mesh
    '''
    def write_3byte_integer(f, n):
        b1 = struct.pack('B', (n >> 16) & 255)
        b2 = struct.pack('B', (n >> 8) & 255)
        b3 = struct.pack('B', (n & 255))
        f.write(b1)
        f.write(b2)
        f.write(b3)


    NEW_VERSION_MAGIC_NUMBER = 16777215
    vnum = len(curv)
    with open(fn, 'wb') as f:
        write_3byte_integer(f, NEW_VERSION_MAGIC_NUMBER)
        f.write(struct.pack(">i", int(vnum)))
        f.write(struct.pack('>i', int(fnum)))
        f.write(struct.pack('>i', 1))
        f.write(curv.astype('>f').tobytes())

def dag_make_overlay_str(**kwargs):        
    masked_data = kwargs.get('masked_data', None)
    cmap = kwargs.get('cmap', 'viridis')    
    if masked_data is not None:
        vmin = kwargs.get('vmin', np.percentile(masked_data, 10))
        vmax = kwargs.get('vmax', np.percentile(masked_data, 90))
    else:
        vmin = kwargs.get('vmin', 0)
        vmax = kwargs.get('vmax', 1)

    cmap_nsteps = kwargs.get('cmap_nsteps', 20)
    
    # Make custom overlay:
    # value - rgb triple...
    fv_param_steps = np.linspace(vmin, vmax, cmap_nsteps)
    fv_color_steps = np.linspace(0,1, cmap_nsteps)
    fv_cmap = dag_cmap_from_str(cmap, **kwargs)
    # fv_cmap = mpl.cm.__dict__[cmap]
    
    ## make colorbar - uncomment to save a png of the color bar...
    # cb_cmap = mpl.cm.__dict__[cmap] 
    # cb_norm = mpl.colors.Normalize()
    # cb_norm.vmin = vmin
    # cb_norm.vmax = vmax
    # plt.close('all')
    # plt.colorbar(mpl.cm.ScalarMappable(norm=cb_norm, cmap=cb_cmap))
    # col_bar = plt.gcf()
    # col_bar.savefig(opj(self.sub_surf_dir, f'lh.{surf_name}_colorbar.png'))

    overlay_custom_str = 'overlay_custom='
    overlay_to_save = '['
    # '''
    # Takes the form 
    # [
    #     {
    #         "r" : 128,
    #         "g" : 0,
    #         "b" : 128.
    #         "val" : -10
    #     },
    #     {
    #         ...
    #     }
    # ]
    # '''
    for i, fv_param in enumerate(fv_param_steps):
        this_col_triple = fv_cmap(fv_color_steps[i])
        this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'
        overlay_custom_str += this_str    
        #
        overlay_to_save += '\n\t{'
        overlay_to_save += f'\n\t\t"b": {int(this_col_triple[2]*255)},'
        overlay_to_save += f'\n\t\t"g": {int(this_col_triple[1]*255)},'
        overlay_to_save += f'\n\t\t"r": {int(this_col_triple[0]*255)},'
        overlay_to_save += f'\n\t\t"val": {float(fv_param):.2f}'
        overlay_to_save += '\n\t}'
        if fv_param!=fv_param_steps[-1]:
            overlay_to_save += ','
    overlay_to_save += '\n]'
    
    return overlay_custom_str, overlay_to_save



# ***********************************************************************************************************************
# STUFF COPIED FROM NIBABEL
# ***********************************************************************************************************************

def dag_serialize_volume_info(volume_info):
    """Copied from from https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py
    Helper for serializing the volume info.
    """
    keys = ['head', 'valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras']
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError(f'Invalid volume info: {diff.pop()}.')

    strings = list()
    for key in keys:
        if key == 'head':
            if not (
                np.array_equal(volume_info[key], [20])
                or np.array_equal(volume_info[key], [2, 0, 20])
            ):
                print('Unknown extension code.')
            strings.append(np.array(volume_info[key], dtype='>i4').tobytes())
        elif key in ('valid', 'filename'):
            val = volume_info[key]
            strings.append(f'{key} = {val}\n'.encode())
        elif key == 'volume':
            val = volume_info[key]
            strings.append(f'{key} = {val[0]} {val[1]} {val[2]}\n'.encode())
        else:
            val = volume_info[key]
            strings.append(f'{key:6s} = {val[0]:.10g} {val[1]:.10g} {val[2]:.10g}\n'.encode())
    return b''.join(strings)

def dag_fread3(fobj):
    """Read a 3-byte int from an open binary file object

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, '>u1', 3)
    return (b1 << 16) + (b2 << 8) + b3


def dag_read_fs_mesh(filepath, return_xyz=False, return_info=True):
    """Adapted from https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py
    ...
    Read a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.

    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    """

    TRIANGLE_MAGIC = 16777214
    with open(filepath, 'rb') as fobj:

        magic = dag_fread3(fobj)
        create_stamp = fobj.readline().rstrip(b'\n').decode('utf-8')
        fobj.readline()
        vnum = np.fromfile(fobj, '>i4', 1)[0]
        fnum = np.fromfile(fobj, '>i4', 1)[0]
        coords = np.fromfile(fobj, '>f4', vnum * 3).reshape(vnum, 3)
        faces = np.fromfile(fobj, '>i4', fnum * 3).reshape(fnum, 3)
        if return_info:
            volume_info = dag_read_volume_info(fobj)        
        else:
            volume_info = {}

    coords = coords.astype(np.float64)  # XXX: due to mayavi bug on mac 32bits

    mesh_info = {
        'vnum' : vnum,
        'fnum' : fnum,
        'coords' : coords,
        'faces' : faces,        
        'volume_info' : volume_info,
    }
    if return_xyz:
        new_mesh_info = {}                                    
        new_mesh_info['x']= mesh_info['coords'][:,0]
        new_mesh_info['y']= mesh_info['coords'][:,1]
        new_mesh_info['z']= mesh_info['coords'][:,2]
        new_mesh_info['i']= mesh_info['faces'][:,0]
        new_mesh_info['j']= mesh_info['faces'][:,1]
        new_mesh_info['k']= mesh_info['faces'][:,2]        
        mesh_info = new_mesh_info

    return mesh_info

def dag_read_volume_info(fobj):
    """Copied from from https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py
    Helper for reading the footer from a surface file.
    """
    volume_info = OrderedDict()
    head = np.fromfile(fobj, '>i4', 1)
    if not np.array_equal(head, [20]):  # Read two bytes more
        head = np.concatenate([head, np.fromfile(fobj, '>i4', 2)])
        if not np.array_equal(head, [2, 0, 20]):
            print.warn('Unknown extension code.')
            return volume_info

    volume_info['head'] = head
    for key in ('valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras'):
        pair = fobj.readline().decode('utf-8').split('=')
        if pair[0].strip() != key or len(pair) != 2:
            raise OSError('Error parsing volume info.')
        if key in ('valid', 'filename'):
            volume_info[key] = pair[1].strip()
        elif key == 'volume':
            volume_info[key] = np.array(pair[1].split(), int)
        else:
            volume_info[key] = np.array(pair[1].split(), float)
    # Ignore the rest
    return volume_info

def dag_serialize_volume_info(volume_info):
    """Copied from from https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py
    Helper for serializing the volume info.
    """
    keys = ['head', 'valid', 'filename', 'volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras']
    diff = set(volume_info.keys()).difference(keys)
    if len(diff) > 0:
        raise ValueError(f'Invalid volume info: {diff.pop()}.')

    strings = list()
    for key in keys:
        if key == 'head':
            if not (
                np.array_equal(volume_info[key], [20])
                or np.array_equal(volume_info[key], [2, 0, 20])
            ):
                print('Unknown extension code.')
            strings.append(np.array(volume_info[key], dtype='>i4').tobytes())
        elif key in ('valid', 'filename'):
            val = volume_info[key]
            strings.append(f'{key} = {val}\n'.encode())
        elif key == 'volume':
            val = volume_info[key]
            strings.append(f'{key} = {val[0]} {val[1]} {val[2]}\n'.encode())
        else:
            val = volume_info[key]
            strings.append(f'{key:6s} = {val[0]:.10g} {val[1]:.10g} {val[2]:.10g}\n'.encode())
    return b''.join(strings)
# ***


def dag_read_fs_curv_file(curv_file):
    with open(curv_file, 'rb') as h_us:
        h_us.seek(15)
        # curv_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
        # Fix updated numpy...
        curv_vals = np.frombuffer(h_us.read(), dtype='>f4').byteswap().view(np.dtype('>f4').newbyteorder())
    return curv_vals

# ***********************************************************************************************

