import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from sklearn.decomposition import PCA
opj = os.path.join

from dpu_mini.utils import *
from dpu_mini.plot_functions import *
from dpu_mini.mesh_format import dpu_cut_box, dpu_mesh_morph, dpu_flatten, dpu_submesh_from_mesh, dpu_ply_write

path_to_utils = os.path.abspath(os.path.dirname(__file__))


class BaseMesh(object):
    '''Generic 2-hemisphere ("lh"/"rh") surface-mesh container.

    Holds mesh coordinates/faces, per-vertex "under surface" values (e.g.,
    curvature-like maps), colour mapping, flattening, submesh extraction,
    and .ply export -- everything GenMeshMaker needs *except* the parts that
    talk to FreeSurfer directly.

    This class has zero dependency on FreeSurfer. Meshes are added directly
    via `add_mesh(mesh_name, hemi, coords, faces)`; there is no notion of a
    FreeSurfer subjects directory here.

    sub         Name/id for this mesh set (used for printing, file naming)
    output_dir  Where to put files we make (e.g., .ply exports)
    '''
    def __init__(self, sub=None, output_dir=None, **kwargs):
        self.sub = sub
        self.output_dir = output_dir
        if isinstance(output_dir, str):
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.do_offsets = kwargs.get('do_offsets', True)
        self.mesh_info = {}
        self.n_vx = {'lh': 0, 'rh': 0}
        self.total_n_vx = 0
        self.n_faces = {'lh': 0, 'rh': 0}
        self.total_n_faces = 0
        self.hemi_masks = {'lh': None, 'rh': None}
        self.us_values = {}
        self.us_cols = {}
        self.ply_files = {}

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region MESH COORDS
    """
    add_mesh            Add a hemisphere mesh directly from coords + faces (no FreeSurfer needed)
    get_mesh_info       Look up a previously added mesh
    idx2bool            Turn a list of vertex indices into a boolean mask
    """
    def add_mesh(self, mesh_name, hemi, coords, faces, do_offsets=None, offset=None, volume_info=None, **kwargs):
        '''Add a hemisphere mesh directly from coordinates + faces.

        mesh_name       str             Name of this mesh (e.g., 'inflated', 'pial', 'sphere', or anything custom)
        hemi            str             'lh' or 'rh'
        coords          np.ndarray      Nx3 vertex coordinates
        faces           np.ndarray      Mx3 face (triangle) indices
        do_offsets      bool            Apply a left/right offset along x, so hemispheres don't overlap
                                         when plotted together (default: self.do_offsets)
        offset          float           Explicit offset to use, overriding the mesh_name-based default
                                         (sphere=100, inflated=50, pial=25, else 0)
        volume_info     dict            Optional FreeSurfer-style header info to carry along (only needed
                                         if you plan to write this mesh back out as a FreeSurfer file)
        '''
        coords = np.asarray(coords, dtype=float)
        faces = np.asarray(faces)
        if do_offsets is None:
            do_offsets = self.do_offsets

        this_mesh_info = {
            'vnum': coords.shape[0],
            'fnum': faces.shape[0],
            'coords': coords.copy(),
            'faces': faces,
            'volume_info': volume_info if volume_info is not None else {},
        }
        x = coords[:, 0].copy()
        if do_offsets:
            if offset is None:
                if 'sphere' in mesh_name:
                    offset = 100
                elif 'inflated' in mesh_name:
                    offset = 50
                elif 'pial' in mesh_name:
                    offset = 25
                else:
                    offset = 0
            x += offset if hemi == 'rh' else -offset
        this_mesh_info['x'] = x
        this_mesh_info['y'] = coords[:, 1]
        this_mesh_info['z'] = coords[:, 2]
        this_mesh_info['i'] = faces[:, 0]
        this_mesh_info['j'] = faces[:, 1]
        this_mesh_info['k'] = faces[:, 2]
        this_mesh_info['coords'] = np.column_stack([this_mesh_info['x'], this_mesh_info['y'], this_mesh_info['z']])

        self.mesh_info.setdefault(mesh_name, {})
        self.mesh_info[mesh_name][hemi] = this_mesh_info

        # Keep vertex/face counts + hemi masks in sync
        self.n_vx[hemi] = coords.shape[0]
        self.n_faces[hemi] = faces.shape[0]
        self.total_n_vx = sum(self.n_vx.values())
        self.total_n_faces = sum(self.n_faces.values())
        self._update_hemi_masks()

        return this_mesh_info

    def _update_hemi_masks(self):
        lhm = np.zeros(self.total_n_vx, dtype=bool)
        rhm = np.zeros(self.total_n_vx, dtype=bool)
        lhm[:self.n_vx['lh']] = True
        rhm[self.n_vx['lh']:] = True
        self.hemi_masks = {'lh': lhm, 'rh': rhm}

    def get_mesh_info(self, key):
        if (key not in self.mesh_info.keys()) or (self.mesh_info[key] == {}):
            raise KeyError(f'No mesh named "{key}" -- add it first with add_mesh(...)')
        return self.mesh_info[key]

    def idx2bool(self, idx):
        b_out = np.zeros(self.total_n_vx, dtype=bool)
        b_out[idx] = True
        return b_out
    #endregion MESH COORDS

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region UNDER SURFACE VALUES
    """
    add_us_values           Add per-vertex "under surface" values directly (e.g., a curvature-like map)
    get_us_values            Fetch previously added under surface values
    get_us_values_per_hemi   Fetch under surface values, split by hemisphere
    """
    def add_us_values(self, key, values_lh, values_rh, **kwargs):
        '''Add per-vertex "under surface" values directly, with zero reliance on FreeSurfer curv files.
        '''
        self.us_values[key] = np.concatenate([values_lh, values_rh])
        if 'cmap' in kwargs:
            self.us_cols[key] = self._data_to_rgb(self.us_values[key], **kwargs)
        return self.us_values[key]

    def get_us_values(self, key, ow=False):
        if (key not in self.us_values.keys()) or ow:
            # raise KeyError(f'"{key}" not found in us_values -- add it first with add_us_values(...)')
            return np.ones(self.total_n_vx)
        return self.us_values[key]

    def get_us_values_per_hemi(self, hemi, key):
        us_vals = self.get_us_values(key).copy()
        if hemi == 'lh':
            us_vals = us_vals[:self.n_vx['lh']]
        elif hemi == 'rh':
            us_vals = us_vals[self.n_vx['lh']:]
        return us_vals
    #endregion UNDER SURFACE VALUES

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region Color functions
    """
    Functions for displaying color on the surface
    return_display_rgb      Return the rgb values for the surface
    _data_to_rgb            Simple mapping from values to colors
    _combine2maps            Combine two maps (e.g., data and undersurface)
    get_us_cols             Get the undersurface colors
    """

    def return_display_rgb(self, data=None, **kwargs):
        '''Per vertex rgb values
            data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.
            under_surf      str             What is going underneath the data (e.g., what is the background)?
                                            default is curv. Could also be thick, (maybe smoothwm)
        **kwargs:
            data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
            data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead
            *** COLOR
            cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                                Default: viridis
            vmin            float           Minimum value for colormap
                                                                Default: min in data
            vmax            float           Max value for colormap
                                                                Default: max in data
            return_cmap_dict bool
        '''
        if not isinstance(data, (np.ndarray, list, pd.DataFrame, pd.Series)):
            print(f'Just using undersurface file..')
            data = np.zeros(self.total_n_vx)
            kwargs['data_alpha'] = np.zeros(self.total_n_vx)

        split_hemi = kwargs.get('split_hemi', False)
        unit_rgb = kwargs.get('unit_rgb', True)
        return_cmap_dict = kwargs.get('return_cmap_dict', False)
        under_surf = kwargs.get('under_surf', 'curv')
        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', np.ones(data.shape[0], dtype=bool))
        data_sub_mask = kwargs.get('data_sub_mask', None)
        data_alpha = kwargs.get('data_alpha', np.ones(data.shape[0]))
        clear_lower = kwargs.get('clear_lower', False)
        clear_upper = kwargs.get('clear_upper', False)

        if data_sub_mask is not None:
            if len(data.shape)==1:
                d_full = np.zeros(self.total_n_vx)
                d_full[data_sub_mask] = data
            else:
                d_full = np.zeros((self.total_n_vx, data.shape[-1]))
                d_full[data_sub_mask,:] = data

            data = d_full
            #
            dm_full = np.zeros(self.total_n_vx, dtype=bool)
            dm_full[data_sub_mask] = data_mask
            data_mask = dm_full
            #
            da_full = np.zeros(self.total_n_vx)
            da_full[data_sub_mask] = data_alpha
            data_alpha = da_full



        data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0

        # ROI arguments
        roi_list = kwargs.get('roi_list', None)
        if roi_list is not None:
            if not isinstance(roi_list, list):
                roi_list = [roi_list]
        roi_fill = kwargs.get('roi_fill', False)
        roi_col = kwargs.get('roi_col', [1,0,0,1])

        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')
        try:
            vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
            vmax = kwargs.get('vmax', np.nanmax(data[data_mask]))
        except:
            vmin = kwargs.get('vmin', np.nanmin(data))
            vmax = kwargs.get('vmax', np.nanmax(data))
        # By default use vmin as mask...
        if clear_lower:
            data_alpha[data<vmin] = 0
        if clear_upper:
            data_alpha[data>vmax] = 0

        # Create rgb values mapping from data to cmap
        data_col1 = self._data_to_rgb(
            data=data, cmap=cmap, vmin=vmin, vmax=vmax
        )
        if data_alpha.mean()==1.0:
            display_rgb = data_col1
        else:
            data_col2 = self.get_us_cols(under_surf)
            display_rgb = self._combine2maps(
                data_col1=data_col1,
                data_col2=data_col2,
                data_alpha=data_alpha)

        # Add ROIs (relies on ROI lookup methods provided by a FreeSurfer-aware subclass)
        if roi_list is not None:
            full_roi_bool = np.zeros(self.total_n_vx, dtype=bool)
            for roi in roi_list:
                if roi_fill:
                    this_roi_bool = self._return_roi_bool_both_hemis(
                        roi_name=roi, return_type='concat',
                    )
                else:
                    this_roi_bool = self._return_roi_bool_border_both_hemis(
                        roi_name=roi, return_type='concat',
                    )
                full_roi_bool[this_roi_bool] += True
            display_rgb[full_roi_bool, :] = roi_col


        if not unit_rgb:
            display_rgb = (display_rgb * 255) # .astype(np.uint8)
        display_rgb_dict = {
            'lh' : display_rgb[:self.n_vx['lh'],:],
            'rh' : display_rgb[self.n_vx['lh']:,:],
        }

        if split_hemi:
            display_rgb = display_rgb_dict
        cmap_dict = {
            'cmap' : cmap,
            'vmin' : vmin,
            'vmax' : vmax,
        }

        if return_cmap_dict:
            return display_rgb, cmap_dict
        else:
            return display_rgb

    def _data_to_rgb(self, data, cmap='viridis', vmin=None, vmax=None):
        '''
        Simple mapping from values to colors
        '''
        # Create rgb values mapping from data to cmap
        if len(data.shape)!=1:
            if data.shape[1] == 3:
                # Assume input is Nx3, add a 4th column of ones
                return np.hstack([data, np.ones((data.shape[0], 1))])
            elif data.shape[1]==4:
                return data
        try:
            data_cmap = dpu_get_cmap(cmap)
        except:
            data_cmap = dpu_cmap_from_str(cmap)
        # data_cmap = dpu_get_cmap(cmap)
        data_norm = mpl.colors.Normalize()
        data_norm.vmin = vmin
        data_norm.vmax = vmax
        data_col = data_cmap(data_norm(data))
        return data_col

    def _combine2maps(self, data_col1, data_col2, data_alpha, **kwargs):
        display_rgb = (data_col1 * data_alpha[...,np.newaxis]) + \
            (data_col2 * (1-data_alpha[...,np.newaxis]))
        return display_rgb

    def get_us_cols(self, key, **kwargs):
        if key not in self.us_cols.keys():
            data = self.get_us_values(key)
            self.us_cols[key] = self._data_to_rgb(
                data, **kwargs
            )
        return self.us_cols[key]

    #endregion Color functions

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region FLAT FUNCTIONS
    def add_flat_to_mesh_info(self, **kwargs):
        '''
        Add the flatmap to the mesh_info
        '''
        flat_name = kwargs.get('flat_name', 'flat')
        flat_pts_hemi = kwargs.get('flat_pts', None)
        flat_polys_hemi = kwargs.get('flat_polys', None)
        vx_to_include = kwargs.get('vx_to_include', np.ones(self.total_n_vx, dtype=bool))
        self.mesh_info[flat_name] = {}
        for hemi in ['lh','rh']:
            flat_pts = flat_pts_hemi[hemi]
            flat_polys = flat_polys_hemi[hemi]
            self.mesh_info[flat_name][hemi] = {}
            self.mesh_info[flat_name][hemi]['coords'] = flat_pts
            self.mesh_info[flat_name][hemi]['faces'] = flat_polys
            self.mesh_info[flat_name][hemi]['x'] = flat_pts[:,0]
            self.mesh_info[flat_name][hemi]['y'] = flat_pts[:,1]
            self.mesh_info[flat_name][hemi]['z'] = flat_pts[:,2]
            self.mesh_info[flat_name][hemi]['i'] = flat_polys[:,0]
            self.mesh_info[flat_name][hemi]['j'] = flat_polys[:,1]
            self.mesh_info[flat_name][hemi]['k'] = flat_polys[:,2]
            # self.mesh_info['']
    def rescale_mesh_info(self, mesh_name):
        for hemi in ['lh', 'rh']:
            flat = self.mesh_info[mesh_name][hemi]
            pts = flat['coords']
            polys = flat['faces']

            connected_pts = np.zeros(len(pts), dtype=bool)
            connected_pts[np.unique(polys)] = True
            pts[connected_pts] -= pts[connected_pts].mean(axis=0)
            try:
                if hemi == 'lh':
                    # Set max x LH to 0
                    pts[connected_pts,0] -= pts[connected_pts,0].max()
                elif hemi == 'rh':
                    # set min x RH to 0
                    pts[connected_pts,0] -= pts[connected_pts,0].min()
            except:
                pass
            self.mesh_info[mesh_name][hemi]['coords'] = pts

    def make_flat_map(self, centre_bool=None, **kwargs):
        '''
        Pycotex uses flatmaps for a bunch of things
        But if you can't be bothered to do it properly, and just want
        to display ROIs in pycortex you can do this

        Custom method to flatten (not using mris_flatten)
        * option 1: use latitude and longitude
        * option 2: do some clever UV mapping with igl code...

        TODO: remove cut from Y
        '''
        out_dir = getattr(self, 'custom_surf_dir', None) or self.output_dir
        if isinstance(out_dir, str) and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        method = kwargs.pop('method', 'sph')
        morph = kwargs.pop('morph', 0) # How much to dilate or erode the mask (if doing igl)
        hemi_project = kwargs.get('hemi_project', 'sphere')
        flat_name = kwargs.get('flat_name', 'flat')
        centre_roi = kwargs.get('centre_roi', None)
        if centre_bool is None:
            centre_bool = np.ones(self.total_n_vx, dtype=bool)
        centre_bool_hemi = {
            'lh': centre_bool[:self.n_vx['lh']],
            'rh': centre_bool[self.n_vx['lh']:]
        }
        vx_to_include = kwargs.pop('vx_to_include', centre_bool)
        vx_to_include = {
            'lh': vx_to_include[:self.n_vx['lh']],
            'rh': vx_to_include[self.n_vx['lh']:]
        }
        cut_box = kwargs.get('cut_box', False)
        cut_circle = kwargs.get('cut_circle', False)

        hemi_pts = {}
        hemi_polys = {}
        pts_combined = []
        polys_combined = []
        # Where to put flatmatp in z plane..
        new_z = np.mean(np.hstack(
            [self.mesh_info[hemi_project]['lh']['z'],self.mesh_info[hemi_project]['rh']['z']]
            ))
        infl_x = np.hstack(
            [self.mesh_info[hemi_project]['lh']['x'],self.mesh_info[hemi_project]['rh']['x']]
            )

        for hemi in ['lh','rh']:
            hemi_kwargs = kwargs.copy()
            hemi_kwargs['z'] = new_z
            # hemi_kwargs['morph'] = morph
            if centre_roi is not None:
                # Load the ROI bool for this hemisphere (needs a FreeSurfer-aware subclass)
                centre_bool_hemi[hemi] |= self._return_roi_bool_both_hemis(centre_roi, **kwargs)[hemi]
            # Cut a box around them?
            # if cut_box:
            #     hemi_kwargs['vx_to_include'] = dpu_cut_box(
            #         mesh_info=self.mesh_info['inflated'][hemi],
            #         vx_bool=centre_bool_hemi[hemi],
            #     )
            elif cut_circle:
                hemi_kwargs['vx_to_include'] = np.ones(self.n_vx[hemi], dtype=bool)
            else:
                hemi_kwargs['vx_to_include'] = vx_to_include[hemi]
            hemi_kwargs['vx_to_include'] = dpu_mesh_morph(
                mesh_info=self.mesh_info[hemi_project][hemi],
                vx_bool=hemi_kwargs['vx_to_include'],
                morph=morph)
            hemi_kwargs['centre_bool'] = centre_bool_hemi[hemi]
            pts,polys,tvx_to_include = dpu_flatten(
                mesh_info=self.mesh_info[hemi_project][hemi],
                method=method,
                **hemi_kwargs)

            flat = pts

            # Demean everything
            connected_pts = np.zeros(len(pts), dtype=bool)
            connected_pts[np.unique(polys)] = True
            flat[connected_pts] -= flat[connected_pts].mean(axis=0)

            # do some cleaning...
            hemi_pts[hemi] = flat.copy()
            hemi_polys[hemi] = polys.copy()
            pts_combined.append(flat)
            if hemi == 'rh':
                polys += len(hemi_pts['lh'])
            polys_combined.append(polys)
        pts_combined[0][:,0] -=pts_combined[0][:,0].max()
        pts_combined = np.vstack(pts_combined)
        polys_combined = np.vstack(polys_combined)
        self.add_flat_to_mesh_info(flat_name=flat_name, flat_pts=hemi_pts, flat_polys=hemi_polys)
        self.rescale_mesh_info(flat_name)

    def flat_mpl(self, **kwargs):
        '''Plot using matplotlib
        '''
        data=kwargs.pop('data', None)
        surf_name = kwargs.pop('surf_name', 'data')
        rot_angles = kwargs.pop('rot_angles', None)
        roi_list = kwargs.pop('roi_list', [])
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        flat_name = kwargs.pop('flat_name', 'flat')
        try:
            self.get_mesh_info(flat_name)
        except:
            self.add_flat_to_mesh_info(flat_name=flat_name)
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
        else:
            fig = ax.get_figure()
        colorbar=kwargs.pop('colorbar', True)

        disp_rgb, cmap_dict = self.return_display_rgb(
            data=data, split_hemi=True, return_cmap_dict=True, **kwargs
        )

        mpts = {
            'lh': self.mesh_info[flat_name]['lh']['coords'],
            'rh': self.mesh_info[flat_name]['rh']['coords'],
        }
        mpolys = {
            'lh': self.mesh_info[flat_name]['lh']['faces'],
            'rh': self.mesh_info[flat_name]['rh']['faces'],
        }
        xlim = [np.inf, -np.inf]
        ylim = [np.inf, -np.inf]
        for hemi in hemi_list:
            if rot_angles is not None:
                mpts[hemi] = dpu_coord_rot(mpts[hemi], rot_angles)
            if np.isnan(mpts[hemi][0][0]):
                continue
            if mpolys[hemi].shape[0]==0:
                continue
            triang = mpl.tri.Triangulation(
                mpts[hemi][:,0],
                mpts[hemi][:,1],
                triangles=mpolys[hemi],
            )
            toincl = np.unique(mpolys[hemi])

            xlim[0] = np.minimum(mpts[hemi][toincl,0].min(), xlim[0])
            xlim[1] = np.maximum(mpts[hemi][toincl,0].max(), xlim[1])
            ylim[0] = np.minimum(mpts[hemi][toincl,1].min(), ylim[0])
            ylim[1] = np.maximum(mpts[hemi][toincl,1].max(), ylim[1])
            # Plot the triangulated data using tripcolor
            cmap = mpl.colors.ListedColormap(disp_rgb[hemi])
            c = np.arange(len(disp_rgb[hemi]))
            ax.tripcolor(
                triang,
                c,
                cmap=cmap,
                shading='gouraud',  # Smooth interpolation between vertices
            )
        for roi in roi_list:
            # Requires a FreeSurfer-aware subclass (ROI outlines come from FS label files)
            roi_obj = self._return_roi_borders_in_order(roi, hemi_list=hemi_list)
            for roi_dict in roi_obj:
                hemi = roi_dict['hemi']
                if hemi not in hemi_list:
                    continue
                x = mpts[hemi][roi_dict['border_vx'],0]
                y = mpts[hemi][roi_dict['border_vx'],1]
                default_settings = {
                        'marker': '.',
                        'color': 'k',
                        'linestyle': 'None',
                        'markersize': 0.8,
                    }
                default_settings.update(kwargs.get('roi_line', {}))
                ax.plot(
                    x,y,
                    **default_settings,
                    label=roi_dict['roi'] if roi_dict['first_instance'] else None,
                )
        # Add color bar
        if colorbar:
            norm = mpl.colors.Normalize(vmin=cmap_dict['vmin'], vmax=cmap_dict['vmax'])
            try:
                cmap = dpu_get_cmap(cmap_dict['cmap'])
            except:
                cmap = dpu_cmap_from_str(cmap_dict['cmap'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, orientation='horizontal', label=surf_name)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.axis('off')
        return cmap_dict

    def reload_flat(self, flat_name):
        '''Reload the flatmap
        '''
        self.mesh_info[flat_name] = {}
        self.add_flat_to_mesh_info(flat_name=flat_name)

    #endregion FLAT FUNCTIONS

    def return_sm(self, mask, hemi, **kwargs):
        surf = kwargs.get('surf', 'pial')
        if mask.shape[0]==self.total_n_vx:
            if hemi=='lh':
                mask = mask[:self.n_vx['lh']]
            elif hemi == 'rh':
                mask = mask[self.n_vx['lh']:]
            else:
                ValueError

        sm = dpu_submesh_from_mesh(
            mesh_info=self.mesh_info[surf][hemi],
            submesh_bool=mask, **kwargs)
        return sm

    def return_pyc_sm(self, mask, hemi, **kwargs):
        sm = self.return_sm(mask, hemi, **kwargs)
        from dpu_mini.pyctx.surface import Surface
        sm = Surface(sm['coords'], sm['faces'])
        return sm

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region PLY FUNCTIONS
    def add_ply_surface(self, data=None, surf_name=None, **kwargs):
        '''
        Arguments:
        Specify surf_name to save
        '''
        overwrite = kwargs.get('ow', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        incl_rgb = kwargs.get('incl_rgb', True)
        if data is None:
            incl_rgb = False
        print(f'File to be named: {surf_name}')
        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))) & (not overwrite) :
            print(f'{surf_name} already exists for {self.sub}, not overwriting surf files...')
            return

        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))):
            print(f'Overwriting: {surf_name} for {self.sub}')
        else:
            print(f'Writing: {surf_name} for {self.sub}')

        # Load mask for data to be plotted on surface
        if incl_rgb:
            display_rgb = self.return_display_rgb(data=data, split_hemi=True, unit_rgb=False, **kwargs)
        else:
            display_rgb = {'lh':None, 'rh':None}

        # Save the mesh files first as .asc, then .srf, then .obj
        # Then save them as .ply files, with the display rgb data for each voxel
        ply_file_2open = []
        for hemi in ['lh', 'rh']:
            ply_str = dpu_ply_write(
                mesh_info   = self.mesh_info[mesh_name][hemi],
                diplay_rgb  = display_rgb[hemi],
                hemi        = hemi,
                values      = data,
                incl_rgb    = incl_rgb,
                )
            ply_surf_file = opj(self.output_dir,f'{hemi}.{surf_name}.ply')
            with open(ply_surf_file, 'w') as file:
                file.write(ply_str)
            ply_file_2open.append(ply_surf_file)
        # Return list of .ply files to open...
        self.ply_files[surf_name] = [
            ply_file_2open
        ]


    #endregion PLY FUNCTIONS
