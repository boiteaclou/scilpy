#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vizualisation for gradient sampling.
Only supports .bvec/.bval and .b (MRtrix).
"""

import argparse
import logging
import numpy as np
import os

from scilpy.utils.bvec_bval_tools import identify_shells
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_gradients_filenames_valid,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.viz.gradient_sampling import (build_ms_from_shell_idx,
                                          load_colors, plot_each_shell,
                                          plot_proj_shell, preload_cusp_cube)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument(
        'gradient_sampling_file', metavar='gradient_sampling_file', nargs='+',
        help='Gradient sampling filename. (only accepts .bvec and .bval '
             'together or only .b).')

    p.add_argument(
        '--dis-sym', action='store_false', dest='enable_sym',
        help='Disable antipodal symmetry.')
    p.add_argument(
        '--out_basename',
        help='Output file name picture without extension ' +
             '(will be png file(s)).')
    p.add_argument(
        '--res', type=int, default=300,
        help='Resolution of the output picture(s).')

    g1 = p.add_argument_group(title='CUSP acquisition parameters')
    g1.add_argument(
        '--cusp', action='store_true', dest='is_cusp',
        help='Specify input as CUSP sampling.')
    g1.add_argument(
        '--b_nominal', type=int, help='Nominal b-value for CUSP sampling.')

    g2 = p.add_argument_group(title='Enable/Disable renderings.')
    g2.add_argument(
        '--dis-sphere', action='store_false', dest='enable_sph',
        help='Disable the rendering of the sphere.')
    g2.add_argument(
        '--dis-proj', action='store_false', dest='enable_proj',
        help='Disable rendering of the projection supershell.')
    g2.add_argument(
        '--plot_shells', action='store_true',
        help='Enable rendering each shell individually.')
    g2.add_argument(
        '--plot_vectors', action='store_true',
        help='Enable rendering of the lines connecting '
             'each vector to the origin.')

    g3 = p.add_argument_group(title='Rendering options.')
    g3.add_argument(
        '--same-color', action='store_true', dest='same_color',
        help='Use same color for all shell.')
    g3.add_argument(
        '--opacity', type=float, default=1.0,
        help='Opacity for the shells.')
    g3.add_argument(
        '--linewidth', type=float, default=2.0,
        help='Width of lines plotted')
    g3.add_argument(
        '--rot_x', type=float, default=0.0,
        help='Camera rotation around focal point with respect to x axis')
    g3.add_argument(
        '--rot_y', type=float, default=0.0,
        help='Camera rotation around focal point with respect to y axis')
    g3.add_argument(
        '--rot_z', type=float, default=0.0,
        help='Camera rotation around focal point with respect to z axis')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.gradient_sampling_file)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if len(args.gradient_sampling_file) == 2:
        assert_gradients_filenames_valid(parser, args.gradient_sampling_file, 'fsl')
    elif len(args.gradient_sampling_file) == 1:
        basename, ext = os.path.splitext(args.gradient_sampling_file[0])
        if ext in ['.bvec', '.bvecs', '.bvals', '.bval']:
            parser.error('You should input two files for fsl format (.bvec '
                         'and .bval).')
        else:
            assert_gradients_filenames_valid(parser, args.gradient_sampling_file, 'mrtrix')
    else:
        parser.error('Depending on the gradient format you should have '
                     'two files for FSL format and one file for MRtrix')

    out_basename = None

    proj = args.enable_proj
    each = args.plot_shells

    if not (proj or each):
        parser.error('Select at least one type of rendering (proj or each).')

    if len(args.gradient_sampling_file) == 2:
        gradient_sampling_files = args.gradient_sampling_file
        gradient_sampling_files.sort()  # [bval, bvec]
        # bvecs/bvals (FSL) format, X Y Z AND b (or transpose)
        points = np.genfromtxt(gradient_sampling_files[1])
        if points.shape[0] == 3:
            points = points.T
        bvals = np.genfromtxt(gradient_sampling_files[0])
        centroids, shell_idx = identify_shells(bvals)
    else:
        # MRtrix format X, Y, Z, b
        gradient_sampling_file = args.gradient_sampling_file[0]
        tmp = np.genfromtxt(gradient_sampling_file, delimiter=' ')
        points = tmp[:, :3]
        bvals = tmp[:, 3]
        centroids, shell_idx = identify_shells(bvals)

    if args.out_basename:
        out_basename, ext = os.path.splitext(args.out_basename)
        possible_output_paths = [out_basename + '_shell_' + str(i) +
                                 '.png' for i in centroids]
        possible_output_paths.append(out_basename + '.png')
        assert_outputs_exist(parser, args, possible_output_paths)

    for b0 in centroids[centroids < 40]:
        shell_idx[shell_idx == b0] = -1
        centroids = np.delete(centroids,  np.where(centroids == b0))

    shell_idx[shell_idx != -1] -= 1
    n_shells = len(np.unique(shell_idx[shell_idx != -1]))

    linewidth = args.linewidth
    rotation = [args.rot_x, args.rot_y, args.rot_z]

    sym = args.enable_sym
    sph = args.enable_sph
    vec = args.plot_vectors
    same = args.same_color
    cusp = args.is_cusp
    cusp_ms, cusp_centroids = None, None

    if cusp:
        cube_centroids_mask = np.logical_and(
            centroids > args.b_nominal, centroids <= 3 * args.b_nominal)
        cube_idx_mask = np.array(
            [s in np.where(cube_centroids_mask)[0] for s in shell_idx])
        cube_idx = np.copy(shell_idx)
        cube_idx[cube_idx_mask] -= cube_idx[cube_idx_mask].min()
        cube_idx[~cube_idx_mask] = -1
        cusp_centroids = []
        for bval in centroids[cube_centroids_mask]:
            centroid_idx = np.where(centroids == bval)[0][0]
            shell_idx[shell_idx == centroid_idx] = -1
            shell_idx[shell_idx > centroid_idx] -= 1
            centroids = np.delete(centroids, np.where(centroids == bval))
            cusp_centroids.append(bval)

        cusp_ms = build_ms_from_shell_idx(points, cube_idx)

    ms = build_ms_from_shell_idx(points, shell_idx)
    if proj:
        scene = None
        load_colors(n_shells)
        if cusp:
            opacity = 0.7 if np.isclose(args.opacity, 1.) else args.opacity
            scene = preload_cusp_cube(
                cusp_ms, np.array(cusp_centroids), args.b_nominal, use_sym=sym,
                use_cube=sph, use_vectors=vec, same_color=same, render=False,
                opacity=opacity, linewidth=linewidth, rotation=rotation,
                ofile=out_basename, ores=(args.res, args.res))

        plot_proj_shell(ms, use_sym=sym, use_sphere=sph, use_vectors=vec,
                        same_color=same, rad=0.025, opacity=args.opacity,
                        linewidth=linewidth, rotation=rotation,
                        scene=scene, ofile=out_basename,
                        ores=(args.res, args.res))
    if each:
        load_colors(n_shells)
        if cusp:
            preload_cusp_cube(
                cusp_ms, np.array(cusp_centroids), args.b_nominal,
                use_sym=sym, use_cube=sph, use_vectors=vec, same_color=same,
                render=True, opacity=args.opacity, linewidth=linewidth,
                rotation=rotation, ofile=out_basename,
                ores=(args.res, args.res))

        plot_each_shell(ms, centroids, plot_sym_vecs=sym, use_sphere=sph,
                        use_vectors=vec, same_color=same, rad=0.025,
                        opacity=args.opacity, linewidth=linewidth,
                        rotation=rotation, ofile=out_basename,
                        ores=(args.res, args.res))


if __name__ == "__main__":
    main()
