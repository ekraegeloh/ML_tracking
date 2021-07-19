#! /usr/bin/python3
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import matplotlib.colors as colors

## ANALYTICAL LIKELIHOOD FUNCTIONS

def llh_scan(pdf_fct, meas, param_table):
    '''
    perform a scan of the analytic likelihood

    Parameters:
    -----------
    pdf_fct     : function which calculates the negative(!) LLH
    meas        : ndarray or tuple; observations
                  if generate_event has more than one output,
                  it needs to be passed as tuple
    param_table : ndarray; parameters
                  columns must correspond to arguments passed to
                  generate_event

    Returns:
    --------
    hit_terms   : ndarray, likelihood values
    '''

    n_params = len(param_table)

    hit_terms = np.empty(n_params)

    for i in range(n_params):
        ps = param_table[i,:]

        if isinstance(meas, tuple):
            hit_terms[i] = pdf_fct(*meas, *ps)
        else:
            hit_terms[i] = pdf_fct(meas, *ps)

    return hit_terms

def map_1d(pdf_fct, meas, ind, steps, base_params):
    '''
    perform a 1-D scan of a parameter of generate_event

    Parameters:
    -----------
    pdf_fct     : function which calculates the negative(!) LLH
    meas        : ndarray or tuple; observations
                  if generate_event has more than one output,
                  it needs to be passed as tuple
    ind         : integer; index of parameter to map
    steps       : ndarray; values of parameter to map
    base_params : ndarray; arguments for generate_event

    Returns:
    --------
    ndarray with results of llh scan
    '''

    n_hypotheses = steps.size

    #in case base_params aren't numpy array yet
    base_params = np.array(base_params)

    #make param table by stacking base params
    param_table = np.repeat(base_params[np.newaxis,:], n_hypotheses, axis=0)
    #replace one argument with map
    param_table[:,ind] = steps

    return llh_scan(pdf_fct, meas, param_table)

def map_2d(pdf_fct, meas, inds, steps, base_params):
    '''
    perform a 2-D scan of a parameter of generate_event

    Parameters:
    -----------
    pdf_fct     : function which calculates the negative(!) LLH
    meas        : ndarray or tuple; observations
                  if generate_event has more than one output,
                  it needs to be passed as tuple
    inds        : tuple of integers; indices of parameters to map
    steps       : tuple of ndarrays; values of parameters to map
    base_params : ndarray; arguments for generate_event

    Returns:
    --------
    ndarray with results of llh scan
    '''

    mg = np.meshgrid(*steps)
    n_hypotheses = mg[0].size

    #in case base_params aren't numpy array yet
    base_params = np.array(base_params)

    #make param table by stacking base params
    param_table = np.repeat(base_params[np.newaxis,:], n_hypotheses, axis=0)
    #replace two arguments with map
    for ind, coord in zip(inds,mg):
        param_table[:,ind] = coord.flat

    return llh_scan(pdf_fct, meas, param_table)


## PLOTTING FUNCTIONS

def colorbar(mappable):
    # from https://joseph-long.com/writing/colorbars/
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    cbar.ax.tick_params(labelsize=12)
    return cbar

def plot_1d_scan(scan, xs, true_x, axis_label, title=None, vmax=None):
    #copy to avoid modifying original scan results
    vals = np.copy(scan)

    #subtract minimum value from scan result
    vals -= vals.min()

    plt.plot(xs, vals, label='scan')
    plt.axvline(x=true_x, color='red', label='truth')

    plt.xlabel(axis_label)
    plt.ylabel('-LLH')
    if title: plt.title(title)
    plt.legend()

def plot_2d_scan(scan, xs, ys, true_x, true_y, axis_labels, title=None,
        vmax=None, log=False):
    #copy to avoid modifying original scan results
    vals = np.copy(scan)

    #subtract minimum value from scan result
    vals -= vals.min()

    if log:
        # make two plots, one linear one log
        fig, ax = plt.subplots(1,2, figsize=(14,5))
        plt.subplots_adjust(wspace=0.5)
    else: fig, ax = plt.subplots()
    gridsize = len(xs)

    if not log: ax = [ax]
    m = ax[0].pcolormesh(xs, ys, vals.reshape(gridsize,gridsize),
            cmap='Spectral', rasterized=True, shading='auto',
            linewidth=0, vmin=0, vmax=vmax)

    if log:
        mlog = ax[1].pcolormesh(xs, ys, vals.reshape(gridsize,gridsize),
                cmap='Spectral', rasterized=True, shading='auto',
                linewidth=0, norm=colors.LogNorm(vmin=1e-3,vmax=vmax))

    for axis in ax:
        #add location of true parameters
        axis.plot([true_x], [true_y], marker='$T$', markersize=10, color='black')
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        if title: axis.set_title(title)
    colorbar(m)
    if log: colorbar(mlog)

# only for analytic and neural net together
def plot_1d_diff(ana, nn, xs, true_x, axis_label, title=None, scale=False):
    #copy to avoid modifying original scan results
    scan_a = np.copy(ana)
    scan_a -= scan_a.min()

    scan_n = np.copy(nn)
    scan_n -= scan_n.min()

    a_label='Analytic'
    if scale:
        scan_a *= scan_n.max()/scan_a.max()
        a_label += ' (scaled)'

    plt.plot(xs, scan_a, label=a_label)
    plt.plot(xs, scan_n, label='Neural net')
    
    #diff = scan_a - scan_b
    #plt.plot(xs, diff, label='difference')

    plt.axvline(x=true_x, color='red', label='truth')
    plt.xlabel(axis_label)
    plt.ylabel('-LLH')
    if title: plt.title(title)
    plt.legend()

def plot_2d_diff(ana, nn, xs, ys, true_x, true_y, axis_labels, title=None,
        vmax=None, vmax_d=None, scale=False, log=False, **kwargs):
    #copy to avoid modifying original scan results
    scan_a = np.copy(ana)
    scan_a -= scan_a.min()

    scan_n = np.copy(nn)
    scan_n -= scan_n.min()

    #resize into 2D
    scan_a = scan_a.reshape((xs.size,ys.size))
    scan_n = scan_n.reshape((xs.size,ys.size))

    a_label = 'Analytic'
    if scale:
        scan_a *= scan_n.max()/scan_a.max()
        a_label += ' (scaled)'

    fig,ax = plt.subplots(1,3, figsize=(23,7))
    plt.subplots_adjust(wspace=0.5)

    if log: norm = colors.LogNorm(vmin=1e-3,vmax=vmax)
    else: norm = colors.Normalize(vmin=0,vmax=vmax)

    m1 = ax[0].pcolormesh(xs, ys, scan_a, cmap='Spectral', rasterized=True,
            shading='auto', linewidth=0, norm=norm, label=r'$\Delta$ LLH', 
            **kwargs)
    ax[0].set_title(a_label)
    colorbar(m1)

    m2 = ax[1].pcolormesh(xs, ys, scan_n, cmap='Spectral', rasterized=True,
            shading='auto', linewidth=0, norm=norm, label=r'$\Delta$ LLH', 
            **kwargs)
    ax[1].set_title('Neural net')
    colorbar(m2)

    diff = scan_a - scan_n
    if not vmax_d: vmax_d = np.max(np.abs(diff))
    md = ax[2].pcolormesh(xs, ys, diff, cmap='RdBu', shading='auto',
            vmin=-vmax_d, vmax=vmax_d, label=r'$\Delta$ LLH', **kwargs)
    ax[2].set_title('Difference')
    colorbar(md)

    for axis in ax:
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        axis.plot([true_x], [true_y], marker='$T$', markersize=10, color='black')


## NEURAL NET FUNCTIONS

def make_dataset(x, t, shuffle_block_size=2**15, batch_size=2**12):
    '''
    get a tensorflow dataset for likelihood approximation

    Parameters:
    -----------
    x                  : ndarray; observations
    t                  : ndarray; parameters        
    shuffle_block_size : int;
                         block size over which to shuffle, 
                         should be multiple of batch_size
    batch_size         : int
        
    Returns:
    --------
        
    tf.data.Dataset
        with structure ((x, t), y) for training
        
    '''
        
    N = x.shape[0]
    assert t.shape[0] == N
        
    d_x = tf.data.Dataset.from_tensor_slices(x)
    d_t = tf.data.Dataset.from_tensor_slices(t)

    d_true_labels = tf.data.Dataset.from_tensor_slices(np.ones((N, 1),
        dtype=x.dtype))
    d_false_labels = tf.data.Dataset.from_tensor_slices(np.zeros((N, 1),
        dtype=x.dtype))

    d_xs = tf.data.Dataset.from_tensor_slices([d_x, 
        d_x]).interleave(lambda x : x)
    d_ts = tf.data.Dataset.from_tensor_slices([d_t, 
        d_t.shuffle(shuffle_block_size)]).interleave(lambda x : x)
    d_ys = tf.data.Dataset.from_tensor_slices([d_true_labels, 
        d_false_labels]).interleave(lambda x : x)
        
        
    dataset = tf.data.Dataset.zip((tf.data.Dataset.zip((d_xs, d_ts)), d_ys))
  
    return dataset.batch(batch_size)
