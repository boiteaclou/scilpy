# -*- coding: utf-8 -*-

import logging
import numpy as np
from tempfile import mkstemp

from dipy.data import get_sphere
from fury import actor, window
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation
import fury

from scilpy.io.utils import snapshot

vtkcolors = [window.colors.blue,
             window.colors.red,
             window.colors.yellow,
             window.colors.purple,
             window.colors.cyan,
             window.colors.green,
             window.colors.orange,
             window.colors.white,
             window.colors.brown,
             window.colors.grey]


def load_colors(N):
    global vtkcolors
    if N > 10:
        vtkcolors = fury.colormap.distinguishable_colormap(
            nb_colors=N, exclude=[(0, 0, 0), (1, 1, 1)])


def get_lines(orig, dest):
    return np.moveaxis([orig, dest], 0, 1)


def rotate_camera(scene, rotation):
    pos, foc, up = scene.get_camera()
    pos = Rotation.from_euler('XYZ', rotation, True).apply(pos)
    up = Rotation.from_euler('XYZ', rotation, True).apply(up)
    scene.set_camera(pos, foc, up)
    return scene


def plot_each_shell(ms, centroids, plot_sym_vecs=True, use_sphere=True,
                    use_vectors=False, same_color=False, rad=0.025,
                    opacity=1.0, linewidth=2.0, rotation=(0., 0., 0.),
                    ofile=None, ores=(300, 300)):
    """
    Plot each shell

    Parameters
    ----------
    ms: list of numpy.ndarray
        bvecs for each bval
    plot_sym_vecs: boolean
        Plot symmetrical vectors
    use_sphere: boolean
        rendering of the sphere
    same_color: boolean
        use same color for all shell
    rad: float
        radius of each point
    opacity: float
        opacity for the shells
    ofile: str
        output filename
    ores: tuple
        resolution of the output png

    Return
    ------
    """
    global vtkcolors

    if use_sphere:
        sphere = get_sphere('symmetric724')
        shape = (1, 1, 1, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        odfs[..., 0] = 1
        affine = np.eye(4)

    for i, shell in enumerate(ms):
        logging.info('Showing shell {}'.format(int(centroids[i])))
        if same_color:
            i = 0
        scene = window.Scene()
        scene.SetBackground(1, 1, 1)
        if use_sphere:
            sphere_actor = actor.odf_slicer(odfs, affine, sphere=sphere,
                                            colormap='winter', scale=1.0,
                                            opacity=opacity)
            scene.add(sphere_actor)
        pts_actor = actor.point(shell, vtkcolors[i], point_radius=rad)
        scene.add(pts_actor)

        if use_vectors:
            vecs_actor = actor.line(
                get_lines(np.zeros_like(shell), shell),
                vtkcolors[i], opacity, linewidth)
            scene.add(vecs_actor)

        if plot_sym_vecs:
            pts_actor = actor.point(-shell, vtkcolors[i], point_radius=rad)
            scene.add(pts_actor)
            if use_vectors:
                vecs_actor = actor.line(
                    get_lines(np.zeros_like(shell), -shell),
                    vtkcolors[i], opacity, linewidth)
                scene.add(vecs_actor)

        scene = rotate_camera(scene, rotation)
        showm = window.ShowManager(scene, order_transparent=True)
        window.show(showm.scene)

        if ofile:
            filename = ofile + '_shell_' + str(int(centroids[i])) + '.png'
            snapshot(showm.scene, filename, size=ores)


def preload_cusp_cube(ms, centroids, b_nominal, use_sym=True, use_cube=True,
                      use_vectors=False, same_color=False, rad=0.025,
                      opacity=0.5, linewidth=2.0, rotation=(0., 0., 0.),
                      render=True, ofile=None, ores=(300, 300)):
    global vtkcolors

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    if render:
        scene = rotate_camera(scene, rotation)

    tetrahedral = np.where(np.isclose(centroids, 3. * b_nominal))[0]
    hexahedral = np.where(np.isclose(centroids, 2. * b_nominal))[0]
    others = np.where(np.logical_and(
        np.logical_and(
            np.greater(centroids, b_nominal),
            np.less(centroids, 3. * b_nominal)),
        np.logical_not(np.isclose(centroids, 2. * b_nominal))))[0]

    for name, proj in [
        (str(3 * b_nominal), tetrahedral),
        (str(2 * b_nominal), hexahedral),
        ("{}_to_{}".format(b_nominal, 3 * b_nominal), others)]:

        for shell in proj:
            pts = np.sqrt(centroids[shell] / b_nominal) * ms[shell]
            if same_color:
                i = 0
            pts_actor = actor.point(pts, vtkcolors[shell], point_radius=rad)
            scene.add(pts_actor)
            if use_vectors:
                vecs_actor = actor.line(
                    get_lines(np.zeros_like(pts), pts),
                    vtkcolors[shell], opacity, linewidth)
                scene.add(vecs_actor)

            if use_sym:
                pts_actor = actor.point(-pts, vtkcolors[shell],
                                        point_radius=rad)
                scene.add(pts_actor)
                if use_vectors:
                    vecs_actor = actor.line(
                        get_lines(np.zeros_like(pts), -pts),
                        vtkcolors[shell], opacity, linewidth)
                    scene.add(vecs_actor)

        if render:
            if use_cube:
                colormap = get_cmap("winter")
                color = colormap(int(colormap.N / 2), alpha=opacity)
                cube = actor.cube(
                    np.array([[0, 0, 0]]), colors=color, scales=[2, 2, 2]
                )
                cube.GetProperty().SetOpacity(opacity)
                scene.add(cube)

            showm = window.ShowManager(scene, order_transparent=True)
            window.show(showm.scene)
            if ofile:
                filename = ofile + '{}_cube.png'.format(name)
                snapshot(showm.scene, filename, size=ores)
            scene.clear()

    if not render and use_cube:
        colormap = get_cmap("winter")
        color = colormap(int(colormap.N / 2), alpha=opacity)
        cube = actor.cube(
            np.array([[0, 0, 0]]), colors=color, scales=[2, 2, 2]
        )
        cube.GetProperty().SetOpacity(opacity)
        scene.add(cube)

    vtkcolors = vtkcolors[len(ms):]

    return scene


def plot_proj_shell(ms, use_sym=True, use_sphere=True, use_vectors=False,
                    same_color=False, rad=0.025, opacity=1.0, linewidth=2.0,
                    rotation=(0., 0., 0.), ofile=None, ores=(300, 300),
                    scene=window.Scene()):
    """
    Plot each shell

    Parameters
    ----------
    ms: list of numpy.ndarray
        bvecs for each bvalue
    use_sym: boolean
        Plot symmetrical vectors
    use_sphere: boolean
        rendering of the sphere
    same_color: boolean
        use same color for all shell
    rad: float
        radius of each point
    opacity: float
        opacity for the shells
    ofile: str
        output filename
    ores: tuple
        resolution of the output png
    scene: vtk.vtkRenderer
        scene preloaded with actors

    Return
    ------
    """
    global vtkcolors

    scene.SetBackground(1, 1, 1)
    if use_sphere:
        sphere = get_sphere('symmetric724')
        shape = (1, 1, 1, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        odfs[..., 0] = 1
        affine = np.eye(4)
        sphere_actor = actor.odf_slicer(odfs, affine, sphere=sphere,
                                        colormap='winter', scale=1.0,
                                        opacity=opacity)

        scene.add(sphere_actor)

    for i, shell in enumerate(ms):
        if same_color:
            i = 0
        pts_actor = actor.point(shell, vtkcolors[i], point_radius=rad)
        scene.add(pts_actor)
        if use_vectors:
            vecs_actor = actor.line(
                get_lines(np.zeros_like(shell), shell),
                vtkcolors[i], opacity, linewidth)
            scene.add(vecs_actor)

        if use_sym:
            pts_actor = actor.point(-shell, vtkcolors[i], point_radius=rad)
            scene.add(pts_actor)
            if use_vectors:
                vecs_actor = actor.line(
                    get_lines(np.zeros_like(shell), -shell),
                    vtkcolors[i], opacity, linewidth)
                scene.add(vecs_actor)

    scene = rotate_camera(scene, rotation)
    showm = window.ShowManager(scene, order_transparent=True)
    window.show(showm.scene)
    if ofile:
        filename = ofile + '.png'
        snapshot(showm.scene, filename, size=ores)


def build_ms_from_shell_idx(bvecs, shell_idx):
    """
    Get bvecs from indexes

    Parameters
    ----------
    bvecs: numpy.ndarray
        bvecs
    shell_idx: numpy.ndarray
        index for each bval

    Return
    ------
    ms: list of numpy.ndarray
        bvecs for each bval
    """

    S = len(set(shell_idx))
    if (-1 in set(shell_idx)):
        S -= 1

    ms = []
    for i_ms in range(S):
        ms.append(bvecs[shell_idx == i_ms])

    return ms
