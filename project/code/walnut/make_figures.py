"""Figures for the stitched Walnut_1 reconstruction."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VOX = 50.0 / 256          # mm per voxel in x/y
VOXZ = 15.0 / 77


def win(a, lo=0.5, hi=99.7):
    return np.percentile(a[np.isfinite(a)], lo), np.percentile(a[np.isfinite(a)], hi)


def main(path='walnut/vol_stitched_hu.npy', tag='stitched'):
    v = np.load(path)
    nx, ny, nz = v.shape
    print(f'{path}: {v.shape}  range [{v.min():.4f},{v.max():.4f}]  '
          f'z extent {nz*VOXZ:.1f} mm')
    lo, hi = win(v)

    # axial montage through the full stitched height
    ks = np.linspace(int(nz*0.06), int(nz*0.94), 12).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, k in zip(axes.ravel(), ks):
        ax.imshow(v[:, :, k].T, cmap='gray', vmin=lo, vmax=hi, origin='lower')
        ax.set_title(f'z = {k*VOXZ:.1f} mm', fontsize=9); ax.axis('off')
    fig.suptitle('Walnut 1 — axial slices through the stitched 4-couch volume '
                 '(Total bin, FDK+hann, CPU port)', fontsize=12)
    fig.tight_layout(); fig.savefig(f'walnut/figs/{tag}_axial.png', dpi=125); plt.close(fig)

    # orthogonal views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(v[:, :, nz//2].T, cmap='gray', vmin=lo, vmax=hi, origin='lower',
                   extent=[0, nx*VOX, 0, ny*VOX])
    axes[0].set_title('axial (mid-height)'); axes[0].set_xlabel('x (mm)'); axes[0].set_ylabel('y (mm)')
    axes[1].imshow(v[:, ny//2, :].T, cmap='gray', vmin=lo, vmax=hi, origin='lower',
                   aspect=VOXZ/VOX, extent=[0, nx*VOX, 0, nz*VOXZ])
    axes[1].set_title('coronal — all 4 couch slabs'); axes[1].set_xlabel('x (mm)'); axes[1].set_ylabel('z (mm)')
    axes[2].imshow(v[nx//2, :, :].T, cmap='gray', vmin=lo, vmax=hi, origin='lower',
                   aspect=VOXZ/VOX, extent=[0, ny*VOX, 0, nz*VOXZ])
    axes[2].set_title('sagittal — all 4 couch slabs'); axes[2].set_xlabel('y (mm)'); axes[2].set_ylabel('z (mm)')
    fig.suptitle('Walnut 1 — stitched reconstruction, orthogonal views', fontsize=12)
    fig.tight_layout(); fig.savefig(f'walnut/figs/{tag}_ortho.png', dpi=130); plt.close(fig)

    # maximum-intensity projection, gives a sense of the whole nut
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(v.max(1).T, cmap='gray', origin='lower', aspect=VOXZ/VOX,
                   extent=[0, nx*VOX, 0, nz*VOXZ])
    axes[0].set_title('MIP along y'); axes[0].set_xlabel('x (mm)'); axes[0].set_ylabel('z (mm)')
    axes[1].imshow(v.max(0).T, cmap='gray', origin='lower', aspect=VOXZ/VOX,
                   extent=[0, ny*VOX, 0, nz*VOXZ])
    axes[1].set_title('MIP along x'); axes[1].set_xlabel('y (mm)'); axes[1].set_ylabel('z (mm)')
    fig.suptitle('Walnut 1 — maximum intensity projections', fontsize=12)
    fig.tight_layout(); fig.savefig(f'walnut/figs/{tag}_mip.png', dpi=130); plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
