"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""
import copy

import h5py
import numpy as np
import matplotlib
from matplotlib import transforms, ticker
from dedalus.core.field import Field
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    # tasks = ['p']
    tasks = ['p_', 'u_', 'w_']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 3, 1
    image = plot_tools.Box(3, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    positions = np.load('rbLocs.npy')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start + count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                if task == 'u' or task == 'u_':
                    if isinstance(dset, Field):
                        dset = plot_tools.FieldWrapper(dset)
                    img_axes = (0, 1, 2)
                    normal_axis = 0
                    image_axes = img_axes[:normal_axis] + img_axes[normal_axis + 1:]
                    image_scales = (0, 0)
                    data_slices = [slice(None), slice(None), slice(None)]
                    data_slices[normal_axis] = index
                    xaxis, yaxis = image_axes
                    xscale, yscale = image_scales
                    xmesh, ymesh, data = plot_tools.get_plane(dset, xaxis, yaxis, data_slices, xscale, yscale)
                    # Setup axes
                    # Bounds (left, bottom, width, height) relative-to-axes
                    pbbox = transforms.Bbox.from_bounds(0.03, 0, 0.94, 0.94)
                    cbbox = transforms.Bbox.from_bounds(0.03, 0.95, 0.94, 0.05)
                    # Convert to relative-to-figure
                    to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
                    pbbox = pbbox.transformed(to_axes_bbox)
                    cbbox = cbbox.transformed(to_axes_bbox)
                    # Create new axes and suppress base axes
                    paxes = axes.figure.add_axes(pbbox)
                    caxes = axes.figure.add_axes(cbbox)
                    axes.axis('off')

                    cmap = 'RdBu_r'
                    cmap = copy.copy(matplotlib.colormaps.get_cmap(cmap))
                    cmap.set_bad('0.7')

                    plot = paxes.pcolormesh(xmesh, ymesh, data, cmap=cmap, zorder=1)
                    paxes.axis(plot_tools.pad_limits(xmesh, ymesh))
                    # paxes.plot(positions)
                    N = 10
                    paxes.plot(positions[0:N, :, 0], positions[0:N, :, 1], '.')
                    paxes.tick_params(length=0, width=0)

                    cbar = plt.colorbar(plot, cax=caxes, orientation='horizontal',
                                        ticks=ticker.MaxNLocator(nbins=5))
                    cbar.outline.set_visible(False)
                    caxes.xaxis.set_ticks_position('top')
                    caxes.set_xlabel(task)
                    caxes.xaxis.set_label_position('top')
                    # paxes.quiver(data[:, :, index], data[:, :, index])
                    # paxes.quiver(data[:, 0, index], data[0, :, index])
                    # paxes.axis(plot_tools.pad_limits(X, Y))
                else:
                    plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True)
                # N = 25
                # axes.plot(positions[N, :, 0], positions[N, :, 1], '.')
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
