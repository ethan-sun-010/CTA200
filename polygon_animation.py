def main():

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
    matplotlib.pyplot.rcParams['image.cmap'] = 'PuBu'
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import pylab
    import glob
    import re
    import numpy
    import h5py
    import tqdm
    import imageio

    def extract_snapshot_number(fpath):

        return int(re.search(r'_(\d+).',fpath).group(1))

    ss_files = sorted(glob.glob('output/snapshot_*.h5'),
                      key=extract_snapshot_number)
    for fname in tqdm.tqdm(ss_files):
        with h5py.File(fname,'r') as f:
            
            vert_x_raw = numpy.array(f['geometry']['x_vertices'])
            vert_y_raw = numpy.array(f['geometry']['y_vertices'])
            vert_n_raw = numpy.array(f['geometry']['n_vertices'])
            vert_idx_list = numpy.concatenate(([0],numpy.cumsum(vert_n_raw))).astype(int)
                    
            polygon_list = [Polygon(
                    numpy.vstack((vert_x_raw[low_idx:high_idx],
                                  vert_y_raw[low_idx:high_idx])).T)
                            for low_idx, high_idx
                            in zip(vert_idx_list[:-1], vert_idx_list[1:])]

            patch_collection = PatchCollection(polygon_list)
            patch_collection.set_array(numpy.log10(f['hydrodynamic']['density']))
            fig, ax = pylab.subplots()
            ax.add_collection(patch_collection)
            pylab.suptitle('t = %.4f' % numpy.array(f['time'])[0])
            pylab.axis('equal')
            pylab.xlim((-2,2))
            pylab.ylim((-2,2))            
            fig.colorbar(patch_collection,ax=ax)
            
            pylab.savefig(fname.replace('.h5','.png'), dpi=240, bbox_inches='tight')

            pylab.close()
            pylab.close(fig)

    with imageio.get_writer('density.mp4',
                            mode='I', fps=30) as writer:
        for fname in tqdm.tqdm(ss_files):
            image = imageio.imread(fname.replace('.h5','.png'))
            writer.append_data(image)

if __name__ == '__main__':

    main()
