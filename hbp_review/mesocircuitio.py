# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 12:24:06 2014
@author: zehl

mesocircuitio
=============
"""

import os
import datetime

import numpy as np
import quantities as pq

import neo
from neo.io.baseio import BaseIO

import util.hdf5_wrapper as h5py

if __name__ == '__main__':
    pass

class MesoCircuitIO(BaseIO):
    """
    Class for reading data in a data file saved from a simulation from the
    Utah Array Model in NEST.
    """
    # Class variables demonstrating capabilities of this IO
    is_readable = True  # This a only reading class
    is_writable = False  # write is not supported

    # This IO can only manipulate continuous data, spikes, and events
    supported_objects = [neo.Block, neo.Segment, neo.SpikeTrain,
                         neo.RecordingChannelGroup, neo.RecordingChannel]

    # Keep things simple by always returning a block
    readable_objects = [neo.Block]

    # And write a block
    writeable_objects = []

    # Not sure what these do, if anything
    has_header = False
    is_streameable = False
    read_params = {}
    write_params = {}

    # The IO name and the file extensions it uses
    name = 'UAModel'
    description = 'This IO reads .h5 file of the Utah Array Model simulated ' + \
        'in NEST.'
    extensions = ['h5']

    # Operates on .h5 files
    mode = 'file'

    def __init__(self, filename, print_diagnostic=False):
        """
        Initialize the UAModel class and associate it to a file.
        """

        self.associated = False
        # Call base constructor
        BaseIO.__init__(self)
        # Remember choice whether to print diagnostic messages or not
        self._print_diagnostic = print_diagnostic
        # Associate to the session
        self._associate(filename=filename)
        # For consistency with baseio
        self.filename = filename

    def _diagnostic_print(self, text):
        '''
        Print a diagnostic message.

        Args:
            text (string):
                Diagnostic text to print.

        Returns:
            -
        '''

        if self._print_diagnostic:
            print('MesoCircuitIO: ' + text)

    def _associate(self, filename):
        """
        Associates the object with a specified session.

        Args:
            filename (string):
                Name of file to associate with.

        Returns:
            -
        """

        # If already associated, disassociate first
        if self.associated:
            raise IOError("Trying to associate an already associated object.")

        # If session name contains a known file extension, remove it
        self.fileformat = filename[-3:]
        self.fileprefix = filename[0:-3]

        # Save session to be associated to
        self.associated_file = filename

        # Create parameter containers
        self.parameters = {}
        self.parameters_electrodes = {}
        self.parameters["Filename"] = self.fileprefix
        self.parameters["Fileformat"] = self.fileformat

        # Does the file exist?
        if not os.path.exists(filename):
            self.file_avail = False
            self._diagnostic_print("No datafile was found for simulation " + \
                filename + ".")
        else:
            self.file_avail = True
            self._diagnostic_print("Scanning " + filename + ".")

            #===================================================================
            # Load metadata
            #===================================================================

            uamodel = {}
            uamodel['metadata'] = h5py.load_h5(filename, path='metadata')

            #===================================================================
            # Scan file
            #===================================================================

            # Number of channels
            self.channel_ids = range(len(
                uamodel['metadata']['electrode_positions_mm'][0]))
            # Get parameterspace_id
            self.parameters['ParameterspaceID'] = \
                uamodel['metadata']['parameterspace_id']
            # Get available layer
            self.parameters['LayersAvailable'] = \
                uamodel['metadata']['layer_names']
            # Date and time of simulation
            self.parameters["DateTime"] = datetime.datetime.strptime(
                uamodel['metadata']['timestamp'], "%Y-%m-%d %H:%M:%S")
            # Simulation time
            self.parameters["SimulationTime"] = \
                uamodel['metadata']['sim_time_ms'] * pq.ms
            self.get_max_time = uamodel['metadata']['sim_time_ms'] * pq.ms
            # Time resolution of simulation (samples/sec)
            self.parameters["TimeResolution"] = int(
                1000. / uamodel['metadata']['time_resolution_ms'])
            self.time_res = int(
                1000. / uamodel['metadata']['time_resolution_ms'])
            self.dt_unit = pq.CompoundUnit("1.0/" + str(self.time_res) + "*s")
            # Neuron densities for loaded layers
            self.parameters["NeuronDensity"] = {}
            for l, lid in enumerate(self.parameters['LayersAvailable']):
                self.parameters["NeuronDensity"][lid] = \
                    uamodel['metadata']['neuron_densities_mm-2'][l] / pq.mm ** 2

            # Electrode parameters
            for chid in self.channel_ids:
                ep = uamodel['metadata']['electrode_positions_mm'][:, chid] * \
                    pq.mm
                es = uamodel['metadata']['electrode_sensitivity_mm'] * pq.mm
                self.parameters_electrodes[chid] = {'Position': ep,
                                                    'Sensitivity': es}

        #=======================================================================
        # Finalize association
        #=======================================================================

        # This object is now successfully associated with a session
        self.associated = True


    def read_block(self, lazy=False, cascade=True, n_starts=[None],
                   n_stops=[None], layer_list=[], channel_list=[], units=[],
                   unit_type=[]):
        """
        TODO: Update docstring
        Reads file contents as a neo Block.

        The Block contains one Segment for each entry in zip(n_starts,
        n_stops). If these parameters are not specified, the default is
        to store all data in one Segment.

        The Block contains one RecordingChannelGroup per channel.

        Args:
            lazy (bool):
                If True, loads the neo block structure without the following data:
                    signal of AnalogSignal objects
                    times of SpikeTrain objects
                    channelindexes of RecordingChannelGroup and Unit objects
            cascade (bool):
                If False, loads only the Block object without children.
            n_starts (list):
                List of starting times as Quantity objects of each Segment, where
                the beginning of the file is time 0. If a list entry is specified as
                None, the beginning of the file is used.
                Default: [None].
            n_stops (list):
                List of corresponding stop times as Quantity objects of each
                Segment. If a list entry is specified as None, the end of the
                file is used (including the last sample). Note that the stop
                time follows pythonic indexing conventions: the sample
                corresponding to the stop time is not returned. Default: [None].
            channel_list (list):
                List of channels to consider in loading. If an empty list is
                specified, all channels are loaded. If None is specified, no
                channels are considered.
                Possible channel IDs: 0 - 99.
                Default: [].
            units (list or dictionary):
                Specifies the unit IDs to load from the data. If an empty list is
                specified, all units of all channels in channel_list are loaded. If
                a list of units is given, all units matching one of the IDs in the
                list are loaded from all channels in channel_list. If a dictionary
                is given, each entry with a key N contains a list of unit IDs to
                load from channel N (but only channels in channel_list are loaded).
                If None is specified, no units are loaded.
                Possible unit IDs: 0 - infinity.
                Default: [].

        Returns:
            neo.Block
                neo Block object containing the data.
                Attributes:
                    name: parameter set ID
                    file_origin: file name
                    rec_datetime: date and time
                    description: string "Utah Array Model format file"

                The neo Block contains the following neo structures:
                Segment
                    For each pair of n_start and n_stop values, one Segment is
                    inserted.
                    Attributes:
                        name: string of the form "Segment i"
                        file_origin: file name
                        rec_datetime: date and time
                        index: consecutive number of the Segment

                RecordingChannelGroup
                    For each recording channel, one RecordingChannelGroup object
                    is created.
                    Attributes:
                        name: string of the form "Channel i"
                        file_origin: session name
                        rec_datetime: date and time
                        channel_indexes: numpy array of one element, which is the integer channel number

                RecordingChannel
                    For each recording channel, one RecordingChannel object is
                    created as child of the respective RecordingChannelGroup.
                    Attributes:
                        name: string of the form "Channel i"
                        file_origin: session name
                        rec_datetime: date and time
                        index: integer channel number

                Unit
                    For each unit, one Unit structure is created.
                    Attributes:
                        name: string of the form "Channel j, Unit u"
                        file_origin: session name
                        channel_indexes: numpy array of one element, which is the integer channel number
                    Annotations:
                        channel_id (int): channel number
                        unit_id (int): unit number

                SpikeTrain
                    For each Unit and each Segment, one SpikeTrain is created.
                    Waveforms of spikes are inserted into the waveforms property of the
                    respective SpikeTrain objects. Individual Spike objects are not
                    used.
                    Attributes:
                        name: string of the form "Segment i, Channel j, Unit u"
                        file_origin: session name
                        dtype: int (original time stamps are save in units of nev_unit)
                        sampling_rate: Waveform time resolution
                    Annotations:
                        channel_id (int): channel number
                        unit_id (int): unit number

            Notes:
                For Segment and SpikeTrain objects, if t_start is not specified, it
                is assumed 0. If t_stop is not specified, the maximum time of all
                analog samples available, all available spike time stamps and all
                trigger time stamps is assumed.
        """

        # Make sure this object is associated
        if not self.associated:
            raise IOError("Cannot load from unassociated parameter set.")

        #=======================================================================
        # Input checking and correcting
        #=======================================================================

        # Correct input for n_starts and n_stops?
        if n_starts is None:
            n_starts = [None]
        elif type(n_starts) == pq.Quantity:
            n_starts = [n_starts]
        elif type(n_starts) != list or \
            any([(type(i) != pq.Quantity and i is not None) for i in n_starts]):
            raise ValueError('Invalid specification of n_starts.')
        if n_stops is None:
            n_stops = [None]
        elif type(n_stops) == pq.Quantity:
            n_stops = [n_stops]
        elif type(n_stops) != list or \
            any([(type(i) != pq.Quantity and i is not None) for i in n_stops]):
            raise ValueError('Invalid specification of n_stops.')

        # Load all layers?
        if layer_list == []:
            # Select all layers
            layer_list = ['L23', 'L4', 'L5', 'L6']
        if type(layer_list) is str:
            # Change format if layer is not given as list
            layer_list = [layer_list]
        if layer_list is None:
            # Select no channel
            layer_list = []
        if type(layer_list) != list:
            raise ValueError('Invalid specification of layer_list.')

        self.parameters["LayersLoaded"] = layer_list


        # Load from all channels?
        if channel_list == []:
            # Select all channels
            channel_list = self.channel_ids
        if channel_list is None:
            # Select no channel
            channel_list = []
        if type(channel_list) != list:
            raise ValueError('Invalid specification of channel_list.')

        # Load all unit types?
        if unit_type == []:
            unit_type = ['excitatory', 'inhibitory']
        elif unit_type == ['excitatory', 'inhibitory']:
            unit_type = unit_type
        elif type(unit_type) is str and \
             unit_type in ['excitatory', 'inhibitory']:
            unit_type = [unit_type]
        else:
            raise ValueError('Invalid specification of unit_type.')

        #=======================================================================
        # Preparations and pre-calculations
        #=======================================================================

        # Lists containing start and stop times (as Quantity) for each segment
        # Used later in creating neo Spiketrain objects
        tstart = []
        tstop = []

        # Create a neo Block
        if self.file_avail:
            # Set time stamp
            recdatetime = self.parameters["DateTime"]
        else:
            # Set time stamp to earliest possible date
            recdatetime = datetime(year=datetime.MINYEAR, month=1, day=1)

        # Create Block
        bl = neo.Block(
            name=os.path.basename(self.associated_file),
            description=self.description,
            file_origin=self.associated_file,
            rec_datetime=recdatetime,
            parameterspace_id=self.parameters['ParameterspaceID'],
            simulation_time=self.parameters['SimulationTime'])

        # Cascade only returns the Block without children, so we are done here
        if not cascade:
            return bl

        # Create segments
        for (sidx, n_start_i, n_stop_i) in \
            zip(range(len(n_starts)), n_starts, n_stops):
            # Make sure start time < end time
            if n_start_i is not  None and n_stop_i is not None and n_start_i >= n_stop_i:
                raise ValueError('An n_starts value is larger than the ' +
                    'corresponding n_stops value.')

            # Define minimum tstart and maximum tstop as start and stop for
            # segment if n_start_i and n_stop_i is None
            if n_start_i is None:
                tstart.append(pq.Quantity(0, self.dt_unit, dtype=int))
            else:
                tstart.append(n_start_i)
            if n_stop_i is None:
                tstop.append(self.get_max_time.rescale(self.dt_unit))
            else:
                tstop.append(n_stop_i)

            # Create Segment for given tstart and tstop
            seg = neo.Segment(
                name="Segment %i" % sidx,
                description='Simulated time from %s to %s' % (
                    str(tstart[-1]), str(tstop[-1])),
                file_origin=self.associated_file,
                rec_datetime=recdatetime,
                index=sidx,
                t_start=tstart[-1],
                t_stop=tstop[-1])

            # Append Segment to Block
            bl.segments.append(seg)

        # Create RecordingChannelGroups, RecordingChannels, Units, Spiketrains
        for lidx, lid in enumerate(self.parameters["LayersLoaded"]):
            if not lazy:
                chids = np.array([ch_i for ch_i in channel_list], dtype=int)
            else:
                chids = np.array([], dtype=int)

            # Create RecordingChannelGroup for each layer
            rcg = neo.RecordingChannelGroup(
                name="Layer %s" % lid,
                description="Layer %s of %s, %s" % (lid,
                    os.path.basename(self.associated_file),
                    self.parameters['ParameterspaceID']),
                channel_indexes=chids,
                channel_names=["Layer %s, Channel %i" % (lid, chid) for \
                    chid in chids],
                file_origin=self.associated_file,
                neuron_density=self.parameters['NeuronDensity'][lid])
            if lazy:
                rcg.lazy_shape = True

            # count through units per channel, but independent of unit_type
            uidx = 0
            for chidx, chid in enumerate(channel_list):
                # Create and append RecordingChannel for each channel in
                # channel_list for each RecordingChannelGroup (layer)
                rcg.recordingchannels.append(
                    neo.RecordingChannel(
                        name="Layer %s, Channel %i" % (lid, chid),
                        index=chid,
                        file_origin=self.associated_file,
                        coordinate=self.parameters_electrodes[chid]['Position'],
                        sensitivity=\
                            self.parameters_electrodes[chid]['Sensitivity'],
                        layer=lid))


                for ut in unit_type:
                    # Load data for each channel
                    path = '/'.join([lid + ut[0].upper(), 'ch%02i' % chid])
                    data = h5py.load_h5(self.associated_file, path=path)

                    # Get units as they are defined
                    all_units = [int(u) for u in data.keys()]
                    if units is None:
                        # Select no units
                        unitids = []
                    elif units == []:
                        # Select all units
                        unitids = all_units
                    elif type(units) == list or type(units) == int:
                        # Select specified units if possible
                        unit_select = units if type(units) is list else [units]
                        unitids = list(set(all_units) & set(unit_select))
                    elif type(units) == dict:
                        # Select units specified for channel
                        if units.has_key(chid):
                            if type(units[chid]) == int:
                                units[chid] = [units[chid]]
                            unitids = list(set(all_units) & set(units[chid]))
                        else:
                            unitids = []
                    else:
                        raise ValueError('Invalid specification of units.')

                    # Create Units and Spiketrains
                    for uid in unitids:
                        # Create Units
                        rcg.units.append(
                            neo.Unit(
                                channel_indexes=chid,
                                name="Channel %i, Unit %i, %s" % (chid, uid, ut),
                                file_origin=self.associated_file,
                                channel_id=chid,
                                unit_id=uid,
                                unit_type=ut))
                        if lazy:
                            rcg.units[uidx].lazy_shape = True

                        # Create spiketrains for each segment
                        for sidx in range(len(bl.segments)):
                            temp = (data['%02i' % uid] * pq.ms).rescale(
                                self.dt_unit)
                            # Mask for getting spikes for spiketrain of segment
                            mask = (((temp >= tstart[sidx]) -
                                (temp < tstop[sidx])) == False)

                            # Create Spiketrain
                            st = neo.SpikeTrain(
                                times=pq.Quantity(temp[mask], self.dt_unit, int),
                                dtype='int',
                                t_start=tstart[sidx],
                                t_stop=tstop[sidx],
                                sampling_rate=self.time_res,
                                name="Segment %i, Channel %i, Unit %i, %s" % (
                                    sidx, chid, uid, ut),
                                file_origin=self.associated_file,
                                segment_id=sidx,
                                channel_id=chid,
                                unit_id=uid,
                                unit_type=ut)
                            if lazy:
                                st.lazy_shape = True

                            # Append spiketrain to RecordingChannelGroup (layer)
                            rcg.units[uidx].spiketrains.append(st)
                            # Create correct relationships
                            rcg.units[uidx].create_relationship()

                            # Append spiketrain to Segment
                            bl.segments[sidx].spiketrains.append(st)
                            # Create correct relationships
                            bl.segments[sidx].create_relationship()

                        # Increase unit count on channel, independent from the
                        # unit_type
                        uidx += 1

            # Create correct relationships
            rcg.create_relationship()

            # Append RecordingChannelGroup (layer) to Block
            bl.recordingchannelgroups.append(rcg)

        # Create correct relationships
        bl.create_relationship()

        return bl


    def save_block(self, block, save_as):
        # Overwrite workaround: removes hdf5-file if it exists
        if os.path.exists(save_as):
            os.remove(save_as)

        # Save Block as hdf5
        hdf_file = neo.io.hdf5io.NeoHdf5IO(filename=save_as)
        hdf_file.save(block)
        hdf_file.close()


    def __str__(self):
        print(self.associated_file)
        print(" ")
        print("File parameters")
        print("====================================")
        print("Time resolution (samples/sec): ", str(self.time_res))
        print("Time unit of simulation: "), str(self.dt_unit)
        print("Available electrode IDs: ", str(self.channel_ids))
        for key_i in self.parameters.keys():
            print(key_i + ": " + str(self.parameters[key_i]))


def tests(filename, save_as):
    obj = MesoCircuitIO(filename, print_diagnostic=True)
    bl = obj.read_block()
    obj.save_block(bl, save_as)

    hdf_file = neo.io.hdf5io.NeoHdf5IO(filename=save_as)
    lbl = hdf_file.get()
    hdf_file.close()

    return obj, bl, lbl

# filename = '/home/zehl/projects/model_NEO-IO/example_data/utah_array_spikes_60s.h5'
# save_as = '/home/zehl/projects/model_NEO-IO/example_data/uaspikes_60s.hdf5'
# obj, bl, lbl = tests(filename, save_as)
