from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
from tqdm import tqdm
import numpy as np
import random
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage.measurements import center_of_mass as cm
import matplotlib.patches as patches
from sep import sum_circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

seed = 42
random.seed(seed)
np.random.seed = 42

# diffractio settings
diameter = 4 * mm
focal = 150 * mm
# defocus = 3 * mm  # distance from focus
defocus = 10 * mm  # distance from focus
wavelength = 0.635 * um
pixels = 50
dx = diameter / pixels  # beam diameter/pixels after cut
dy = dx
cut_field = 400
image_size = 128
# generate source and lense
x0 = np.linspace(-diameter / 2, diameter / 2, pixels)  # note: size of phase frame is 256x256
y0 = np.linspace(-diameter / 2, diameter / 2, pixels)
u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)  # initial beam - planar wave
t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)  # create lens
t0.lens(r0=(0 * um, 0 * um),  # t0 - field on lens
        radius=(diameter / 2, diameter / 2),
        focal=(focal, focal))
# zernike pyramid for n (radial degree)
n_values = [0,
            1, 1,
            2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6]

# zernike pyramid for m (azimuthal degree)
m_values = [0,
            -1, 1,
            -2, 0, 2,
            -3, -1, 1, 3,
            -4, -2, 0, 2, 4,
            -5, -3, -1, 1, 3, 5,
            -6, -4, -2, 0, 2, 4, 6]

### list all data phase files ###
data_Mfolder = r'D:\high_r0_phase_database'
data_dirs = glob.glob(data_Mfolder + r'\*')
phase_paths = []
for directory in data_dirs:
    phase_paths.append(glob.glob(directory + r'\*.npy'))
phase_paths = np.reshape(phase_paths, [-1])
phase_paths.tolist()

for i, path in tqdm(enumerate(phase_paths[100:110])):
    # returned_zernike_coeffs = []
    # # IMPORTANT note: phase movie is 256x256 pixels, 1x1m, and we use 20cm pupil
    # phase_frame = np.load(path)[103:153, 103:153]  # cuts 50x50 pixels from center of phase frame
    # # for j in range(len(m_values)):
    # temp_u0 = u0
    # # temp_coeffs = np.zeros(len(m_values))
    # # temp_coeffs[1] = 1
    # temp_u0.zernike_beam(A=1,
    #                      r0=(0, 0),
    #                      radius=diameter / 2,
    #                      n=[1,1],
    #                      m=[-1,1],
    #                      c_nm = [15,15])
                         #
                         # n=n_values,
                         # m=m_values,
                         # c_nm=temp_coeffs)
    # returned_zernike_coeffs.append((1 / (diameter / 2) ** 2) *
    #                            np.sum(phase_frame * np.angle(temp_u0.u) * dx * dy))
    # u0.zernike_beam(A=1,
    #                 r0=(0, 0),
    #                 radius=diameter / 2,
    #                 n=n_values,
    #                 m=m_values,
    #                 c_nm=returned_zernike_coeffs)
    # u0 = temp_u0
    u1 = u0 * t0  # field right ater lense

    # de focus plane
    u2 = u1.RS(z=focal - defocus)  # reileigh somerfeld from lense to first camera
    u2.cut_resample(x_limits=(-cut_field, cut_field),
                    y_limits=(-cut_field, cut_field),
                    num_points=(image_size, image_size),
                    new_field=False,
                    interp_kind=(3, 1))
    image_defocus = np.abs(u2.u) ** 2  # image in first camera (defocused image)
    # focus plane
    u3 = u1.RS(z=focal)  # reileigh somerfeld from lense to second camera
    u3.cut_resample(x_limits=(-cut_field, cut_field),
                    y_limits=(-cut_field, cut_field),
                    num_points=(image_size, image_size),
                    new_field=False,
                    interp_kind=(3, 1))
    image_focus = np.abs(u3.u) ** 2  # image in second camera (focused image)

    ### PLOTS ###
    #
    # u0.draw("phase")
    # u1.draw('field')
    # plt.show()
    u2.draw('field')
    plt.show()
    u3.draw('field')
    plt.show()
    dat_defocus = np.abs(u2.u)**2
    dat_focus = np.abs(u3.u)**2
    to_cut = 20
    # x_cm, y_cm = 64, 64
    y_cm, x_cm = cm(dat_focus)

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 5))
    # ax0.imshow(np.abs(u2.u)**2)
    # ax0.set_title("Intensity, defocus plane")
    # ax1.imshow(np.abs(u3.u)**2)
    # ax1.set_title("Intensity, focus plane")
    # plt.show()

    ### Plot original spot at FOCUS ###
    dat_focus = dat_focus.copy(order='C')  # solves "arry not C-contiguous error"
    im0 = ax0.imshow(dat_focus)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)
    intensity_in_circle = sum_circle(dat_focus, [x_cm], [y_cm], [to_cut])[0]
    intensity_in_circle = intensity_in_circle / np.sum(dat_focus)
    pupil = plt.Circle((x_cm, y_cm), to_cut, color='red', fill=False)  # Create a circle patch
    ax0.add_patch(pupil)
    ax0.set_title(f'Spot in focus, relative intensity in window={float(np.around(intensity_in_circle, 3))}')

    ### Plot original spot at DEFOCUS ###
    dat_defocus = dat_defocus.copy(order='C')  # solves "arry not C-contiguous error"
    im1 = ax1.imshow(dat_defocus)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)
    intensity_in_circle = sum_circle(dat_defocus, [x_cm], [y_cm], [to_cut])[0]
    intensity_in_circle = intensity_in_circle / np.sum(dat_defocus)
    pupil = plt.Circle((x_cm, y_cm), to_cut, color='red', fill=False)  # Create a circle patch
    ax1.add_patch(pupil)
    ax1.set_title(f'Spot in defocus, relative intensity in window={float(np.around(intensity_in_circle, 3))}')

    fig.tight_layout()
    print(cm(dat_focus))
    plt.show()


    # fig, [ax0,ax1] = plt.subplots(1,2, figsize=(10,5))
    # im0 = ax0.imshow(phase_frame)
    # divider = make_axes_locatable(ax0)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # ax0.set_title("original phase, cut")
    # fig.colorbar(im0,cax=cax)
    # im1 = ax1.imshow(np.angle(u0.u))
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # ax1.set_title("Reconstructed phase")
    # fig.colorbar(im1,cax=cax)
    # fig.tight_layout()

    # bundle data to be saved later
    frame = np.zeros((image_size, image_size, 2))
    frame[:, :, 0] = image_focus / np.max(image_focus)  # fill in the first channel, normalized
    frame[:, :, 1] = image_defocus / np.max(image_defocus)  # fill in the second channel, normalized
    label = returned_zernike_coeffs
    data_frame = [frame, label]  # bundle together each frame with its label