import os
import glob

import numpy as np
import quantities as pq
from hbp_review_task_collect import num_neurons


#==============================================================================
# HDF5 load code
#==============================================================================

# Auxiliary functions

def delete_group(f, group) :
    try:
        f = h5py.File(f, 'r+')
        try:
            del f[group]
            f.close()
        except KeyError:
            f.close()
    except IOError:
        pass

def node_exists(f, key):
    f = h5py.File(f, 'r')
    exist = key in f
    f.close()
    return exist


def dict_to_h5(d, f, overwrite_dataset, compression=None, **keywords) :
    if 'parent_group' in keywords :
        parent_group = keywords['parent_group']
    else:
        parent_group = f.parent

    for k, v in d.items() :
        if isinstance(v, collections.MutableMapping) :
            if parent_group.name != '/':
                group_name = parent_group.name + '/' + str(k)
            else:
                group_name = parent_group.name + str(k)
            group = f.require_group(group_name)
            dict_to_h5(v, f, overwrite_dataset, parent_group=group, compression=compression)
        else:
            if not str(k) in parent_group.keys() :
                create_dataset(parent_group, k, v, compression=compression)
            else:
                if overwrite_dataset == True:
                    del parent_group[str(k)]  # delete the dataset
                    create_dataset(parent_group, k, v, compression=compression)
                else:
                    print 'Dataset', str(k), 'already exists!'
    return 0  # ?


def create_dataset(parent_group, k, v, compression=None):
    shp = numpy.shape(v)
    if v == None:
        parent_group.create_dataset(str(k), data='None', compression=compression)
    else:
        if isinstance(v, (list, numpy.ndarray)):
                if numpy.array(v).dtype.name == 'object':
                    if len(shp) > 1:
                        print 'Dataset', k, 'has an unsupported format!'
                    else:
                        # store 2d array with an unsupported format by reducing
                        # it to a 1d array and storing the original shape
                        # this does not work in 3d!
                        oldshape = numpy.array([len(x) for x in v])
                        data_reshaped = numpy.hstack(v)
                        data_set = parent_group.create_dataset(str(k), data=data_reshaped, compression=compression)
                        data_set.attrs['oldshape'] = oldshape
                        data_set.attrs['custom_shape'] = True
                elif isinstance(v, pq.Quantity) :
                    data_set = parent_group.create_dataset(str(k), data=v)
                    data_set.attrs['_unit'] = v.dimensionality.string
                else :
                    data_set = parent_group.create_dataset(str(k), data=v, compression=compression)
        elif isinstance(v, (int, float)) :  # ## ignore compression argument for scalar datasets
            data_set = parent_group.create_dataset(str(k), data=v)
        else:
            data_set = parent_group.create_dataset(str(k), data=v, compression=compression)

        # ## Explicitely store type of key
        _key_type = type(k).__name__
        data_set.attrs['_key_type'] = _key_type


def dict_from_h5(f):
    # .value converts everything(?) to numpy.arrays
    # maybe there is a different option to load it, to keep, e.g., list-type
    if h5py.h5i.get_type(f.id) == 5:  # check if f is a dataset
        if hasattr(f, 'value'):
            if 'EMPTYARRAY' in str(f.value):  # ## This if-branch exists to enable loading of deprecated hdf5 files
                shp = f.value.split('_')[1]
                shp = tuple(int(i) for i in shp[1:-1].split(',') if i != '')
                return numpy.reshape(numpy.array([]), shp)
            elif str(f.value) == 'None':
                return None
            else:
                if len(f.attrs.keys()) > 0 and 'custom_shape' in f.attrs.keys() :
                    if f.attrs['custom_shape']:
                        return load_custom_shape(f.attrs['oldshape'], f.value)
                else:
                    return f.value
        else:
            return numpy.array([])
    else:
        d = {}
        items = f.items()
        for name, obj in items :
            if h5py.h5i.get_type(obj.id) == 2 :  # Check if obj is a group or a dataset
                sub_d = dict_from_h5(obj)
                d[name] = sub_d
            else :
                if hasattr(obj, 'value'):
                    if 'EMPTYARRAY' in str(obj.value):
                        shp = obj.value.split('_')[1]
                        shp = tuple(int(i) for i in shp[1:-1].split(',') if i != '')
                        d[name] = numpy.reshape(numpy.array([]), shp)
                    elif str(obj.value) == 'None':
                        d[name] = None
                    else:
                        # if dataset has custom_shape=True, we rebuild the original array
                        if len(obj.attrs.keys()) > 0 :
                            if 'custom_shape' in obj.attrs.keys() :
                                if obj.attrs['custom_shape']:
                                    d[name] = load_custom_shape(obj.attrs['oldshape'], obj.value)
                            elif '_unit' in obj.attrs.keys() :
                                d[name] = pq.Quantity(obj.value, obj.attrs['_unit'])
                            elif '_key_type' in obj.attrs.keys() :
                                # added string_ to handle numpy.string_, TODO: find general soluation for numpy data types
                                if obj.attrs['_key_type'] not in ['str', 'unicode', 'string_'] :
                                    d[ast.literal_eval(name)] = obj.value
                                else :
                                    d[name] = obj.value
                        else:
                            d[name] = obj.value
                else:
                    d[name] = numpy.array([])
        return d

def load_custom_shape(oldshape, oldata):
    data_reshaped = []
    counter = 0
    for l in oldshape:
        data_reshaped.append(numpy.array(oldata[counter:counter + l]))
        counter += l
    return numpy.array(data_reshaped, dtype=object)


# Save routine
def add_to_h5(filename, d, write_mode='a', overwrite_dataset=False, resize=False, dict_label='', compression=None) :
    '''
    Save dictionary containing data to hdf5 file.

    **Args**:
        filename: file name of the hdf5 file to be created
        d: dictionary to be stored
        write_mode: can be set to 'a'(append) or 'w'(overwrite), analog to normal file handling in python (default='a')
        overwrite_dataset: whether all datasets should be overwritten if already existing. (default=False)
        resize: if True, the hdf5 file is resized after writing all data, may reduce file size, caution: slows down writing (default=False)
        dict_label: If given, the dictionary is stored as a group with the given name in the hdf5 file, labels can also given as paths to target lower levels of groups, e.g.: dict_label='test/trial/spiketrains' (default='')
        compression: Compression strategy to reduce file size.  Legal values are 'gzip', 'szip','lzf'.  Can also use an integer in range(10) indicating gzip, indicating the level of compression. 'gzip' is recommended. Caution: This slows down writing and loading of data. Attention: Will be ignored for scalar data.

    '''
    try:
        f = h5py.File(filename, write_mode)
    except IOError:
        raise IOError('unable to create ' + filename + ' (File accessability: Unable to open file)')
    if dict_label != '' :
        base = f.require_group(dict_label)
        dict_to_h5(d, f, overwrite_dataset, parent_group=base, compression=compression)
    else:
        dict_to_h5(d, f, overwrite_dataset, compression=compression)
    fname = f.filename
    f.close()
    if overwrite_dataset == True and resize == True:
        call(['h5repack', '-i', fname, '-o', fname + '_repack'])
        call(['mv', fname + '_repack', fname])
    return 0

# Load routine
def load_h5(filename, path='') :
    '''
    The Function returns a dictionary of all dictionaries that are
    stored in the HDF5 File.

    **Args**:
        filename: file name of the hdf5 file to be loaded
        path: argument to access deeper levels in the hdf5 file (default='')
    '''

    d = {}
    try:
        f = h5py.File(filename, 'r')
    except IOError:
        raise IOError('unable to open \"' + filename + '\" (File accessability: Unable to open file)')
    if path == '':
        d = dict_from_h5(f)
    else:
        if path[0] == '/':
            path = path[1:]
        if node_exists(filename, path):
            d = dict_from_h5(f[path])
        else:
            f.close()
            raise KeyError('unable to open \"' + filename + '/' + path + '\" (Key accessability: Unable to access key)')
    f.close()
    return d


#==============================================================================
# Loading starts here
#==============================================================================

filename = '../results/hbp_review_task/viz_output_mdl.h5'
cc = load_h5(filename)

# example: build correlation matrix
num_neurons = cc['meta']['num_neurons']
C = np.zeros((num_neurons, num_neurons))
for edge_i, p_i in cc['func_conn']['cch']['pvalue']:
    x = cc['edges']['id_i'][edge_i]
    y = cc['edges']['id_j'][edge_i]
    C[x, y] = C[y, x] = p_i

# example: build 10x10 matrix of firing rates (assuming one neuron per x,y coordinate)
T = np.zeros((10, 10))
for neuron_i in range(num_neurons):
    x = cc['neuron_topo']['x'][neuron_i]
    y = cc['neuron_topo']['y'][neuron_i]
    T[x, y] = cc['neuron_single_values']['rate'][neuron_i]

# cc = {}
#
# cc['meta'] = {}
# cc['meta']['num_neurons'] <- one value
# cc['meta']['num_edges'] <- one value
#
# cc['neuron_topo'] = {}
# cc['neuron_topo']['x'] = {} <- one entry for each neuron ID
# cc['neuron_topo']['y'] = {} <- one entry for each neuron ID
# cc['neuron_topo']'behavioral']['c'] = {} <- one entry for each neuron ID
#
# cc['func_conn'] = {}
# cc['func_conn']['cch_peak'] = {}
# cc['func_conn']['cch_peak']['pvalue'] = {} <- one entry for each edge
#
# cc['edges'] = {}
# cc['edges']['id_i'] = {} <- one entry for each edge
# cc['edges']['id_j'] = {} <- one entry for each edge
#
# cc['neuron_single_values'] = {}
# cc['neuron_single_values']['rate'] = {} <- one entry for each neuron ID
# cc['neuron_single_values']['cv'] = {} <- one entry for each neuron ID
# cc['neuron_single_values']['lv'] = {} <- one entry for each neuron ID
# cc['neuron_single_values']['behavior'] = {} <- one entry for each neuron ID
#
# cc['edge_time_series'] = {}
# cc['edge_time_series']['cch'] = {} <- one entry for each edge
# cc['edge_time_series']['sig_upper_975'] = {} <- one entry for each edge
# cc['edge_time_series']['sig_lower_25'] = {} <- one entry for each edge
#

