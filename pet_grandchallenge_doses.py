import glob
import matplotlib.pyplot as plt
from pydicom import dcmread
from PIL import Image
import re
from skimage.restoration import denoise_wavelet
from skimage import img_as_float
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from pydicom.dataset import FileDataset, FileMetaDataset
from sklearn.metrics import mean_squared_error
import datetime
import os
import tempfile
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from common.utils import apply_random_mask, psnr, load_image, print_progress, print_end_message, print_start_message
from common.operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox, norm1, norm2sq
from skimage.metrics import structural_similarity as ssim


def ISTA(fx, gx, gradf, proxg, params, verbose = False):
    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        y = x_k - alpha * gradf(x_k)
        x_k_next = proxg(y, alpha * lmbd)
        x_k = x_k_next

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0 and verbose:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
    '''
    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
        '''
    method_name = 'FISTA' # added
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']
    y_k = x0
    t_k = 1
    restart_fista = params['restart_criterion']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update iterate
        prox_argument = y_k - alpha * gradf(y_k)
        x_k_next = proxg(prox_argument, alpha)
        t_k_next = (1 + np.sqrt(4 * (t_k ** 2) + 1)) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        if restart_fista and gradient_scheme_restart_condition(x_k.reshape(x_k.shape[0],), x_k_next.reshape(x_k_next.shape[0],), y_k.reshape(y_k.shape[0],)):
            y_k = x_k
        else:
            y_k = y_k_next
            t_k = t_k_next
            x_k = x_k_next


        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            if verbose:
                print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    if verbose:
        print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0

def reconstructL1(image, indices, optimizer, params):
    # Wavelet operator
    r = RepresentationOperator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices)  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x,indices,params['m']))  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image, indices)

    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x:  norm1(x)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, info = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstructTV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: p_omega(x,indices)  # P_Omega
    adjoint_operator = lambda x: p_omega_t(x,indices, params['m']) # P_Omega^T

    # Generate measurements
    b = forward_operator(image)

    fx = lambda x: norm2sq(b - forward_operator(x))
    gx = lambda x: TV_norm(x, optimizer)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(x.shape[0],1)

    x, info = optimizer(fx, gx, gradf, proxg, params, verbose=params['verbose'])
    return x.reshape((params['m'], params['m'])), info


print("Starting the training of the files")
files = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-100 dose/*.IMA")


dose12 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-2 dose/*.IMA")
dose14 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-4 dose/*.IMA")
dose110 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-10 dose/*.IMA")
dose120 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-20 dose/*.IMA")
dose150 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-50 dose/*.IMA")
dose1100 = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/1-100 dose/*.IMA")

ref_files = glob.glob("./data/training/Siemens Vision Quadra-*/Siemens Vision Quadra/*/*/Full_dose/*.IMA")

print(len(ref_files))
print(len(dose150))



shape = (256, 256)
params = {
        'maxit': 200,
        'tol': 10e-15,
        'prox_Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': True,
        'stopping_criterion': False,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }

print("Training Completed: Starting testing")

#all_images=glob.glob('./data/test/*/*.nii.gz')
all_images=glob.glob('./data/test/020822_1_20220802_171408/DRF_100.nii.gz')


files = [os.path.basename(x) for x in all_images]

path = [os.path.dirname(x) for x in all_images]

param =[]

counter=0

filtered = [fn for fn in all_images 
         if not os.path.basename(fn).startswith("new")]



for test_files in filtered:
    test_image=nib.load(test_files).get_fdata()
    u, s, vh = np.linalg.svd(test_image, full_matrices=False)
    
    wavelet_smoothed = denoise_wavelet(test_image, multichannel=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True,wavelet='db2')


    img_denoised = denoise_tv_chambolle(wavelet_smoothed, weight=3, multichannel=False)
    noise_psnr = peak_signal_noise_ratio(test_image, img_denoised, data_range=1.0)
    print("PSNR of input noisy image = ", noise_psnr)
    plt.imshow(img_denoised[img_denoised.shape[0]//2])
    plt.savefig('books_read.png')
    #denoise_TV = test_image
    img = nib.Nifti1Image(img_denoised, np.eye(4))
    img.get_data_dtype() == np.dtype(np.int16)
    img.header.get_xyzt_units()
    nib.save(img, os.path.join(path[counter], "new"+files[counter])) 
    counter=counter+1

print("Closing")
