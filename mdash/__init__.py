from collections import defaultdict, namedtuple
import numpy as np
import re


class DashOutput(object):
    def __init__(self, dash_output_file, dash_input_file=None):
        self.dihedrals = []
        self.states = []
        self.state_trajectory = []
        self.options = {}
        self.NullState = None
        while True:
            title, block = self._get_next_block(dash_output_file)
            if block is None:
                break
            if title == 'OPTIONS':
                for line in block:
                    key, value = line.split(':')
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip()
                    self.options[key.strip()] = value
            elif re.match('ANGLE\_[0-9]+', title):
                self.dihedrals.append(Dihedral(title, block))
            elif re.match('(REPLICA_EXCHANGE|DASH)_STATES', title):
                for line in block:
                    self.states.append(State(line, self.dihedrals))
            elif '_STATE_DISTRIBUTION' in title:
                for line, state in zip(block[1:], self.states):
                    state._add_rep_frame(line)
            elif '_STATE_TRAJECTORY' in title:
                self._create_null_state()

                for line in block[1:]:
                    cols = list(map(int, line.split()))
                    for i in range(cols[1]):
                        state = self.get_state(cols[0])
                        for dihedral_state, dihedral in zip(
                                state.dihedral_states, self.dihedrals):
                            dihedral.state_trajectory.append(dihedral_state)
                        self.state_trajectory.append(state)
            elif '_STATE_MEAN' in title:
                for line, state in zip(block, self.states):
                    state._add_mean_angles(line)
            elif '_STATE_STANDARD_DEVIATIONS' in title:
                for line, state in zip(block, self.states):
                    state._add_stdev(line)
            elif '_STATE_CIRCULAR_SIMILARITY' in title:
                for line, state in zip(block[1:], self.states):
                    state._add_circular_similarity(line, self.states)
            elif '_STATE_COSINE_SIMILARITY' in title:
                for line, state in zip(block[1:], self.states):
                    state._add_cosine_similarity(line, self.states)

        if dash_input_file is not None:
            self.trajectory = np.loadtxt(dash_input_file)
            for traj, dihedral in zip(self.trajectory.T, self.dihedrals):
                dihedral.trajectory = traj

    def _get_next_block(self, f):
        """Return the next set of lines corresponding to an entry in the dash
        output files."""
        lines = []
        for line in f:
            if line.startswith('['):
                title = line[1:-2]
                break
        else:
            return None, None

        for line in f:
            if line == '\n':
                break
            else:
                lines.append(line)
        return title, lines

    def get_state(self, index=None):
        """Return the State object corresponding to the specified criteria"""
        if index == 0:
            return self.NullState
        for state in self.states:
            if state.index == index:
                return state
        raise ValueError(
            "Unable to find state object with values: index=%s" % index)

    def _create_null_state(self):
        # Need to create a NullState
        NullDihedralState = DihedralState(None, None, None)
        self.NullState = State('0', [])
        self.NullState.dihedral_states = [NullDihedralState] * self.n_dihedrals
        self.NullState.mean_angles = [None] * self.n_dihedrals
        self.NullState.stdev_angles = [None] * self.n_dihedrals
        self.NullState.index

    def combine_states(self, indices):
        """Within the system state trajectory combine together the specified
        states. This can be handy to """
        i = indices[0]
        state = self.get_state(index=i)
        for index in indices[1:]:
            state_to_remove = self.get_state(index=index)
            self.states.remove(state_to_remove)
            for i, tstate in enumerate(self.state_trajectory):
                if tstate is state_to_remove:
                    self.state_trajectory[i] = state
        self.reindex_by_frequency()

    def reindex_by_frequency(self):
        new_order = sorted(
            self.states,
            key=lambda x: self.state_trajectory.count(x), reverse=True)
        for i, state in enumerate(new_order, 1):
            state.index = i

    def combine_states_by_similarity(self, threshold):
        i = 1
        while True:
            if i > len(self.states):
                break
            state = self.get_state(index=i)
            to_combine = [i]
            for other_state in self.states[i:]:
                if state.cosine_similarity[other_state] >= threshold:
                    to_combine.append(other_state.index)
            self.combine_states(to_combine)
            i += 1

    @property
    def n_dihedrals(self):
        return len(self.dihedrals)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_frames(self):
        return len(self.state_trajectory)

    def reindex_by_occurence(self):
        """Regassign indices to state objects such that they match the order
        in which they arise in the system state trajectory
        """
        index_map = {}
        count = 1
        for state in self.state_trajectory:
            if state is self.NullState:
                continue
            try:
                state.index = index_map[state]
            except KeyError:
                index_map[state] = count
                state.index = count
                count += 1


class State(object):
    """This class describes all of the relevant information corresponding
    to a state of the system i.e. a unique combination of the obseverved
    dihedral values during the simulation"""
    def __init__(self, line, dihedrals):
        cols = line.split()
        self.index = int(cols[0])
        self.dihedral_states = [dih.get_state(index=int(i))
                                for dih, i in zip(dihedrals, cols[1:])]
        self.mean_angles = np.array([])
        self.stdev_angles = np.array([])
        self.rep_frame = None
        self.rep_frame_rmsd = 0.
        self.circular_similarity = {}
        self.cosine_similarity = {}

    @property
    def state_code(self):
        return "".join(str(dih.index) for dih in self.dihedral_states)

    def _add_mean_angles(self, line):
        self.mean_angles = np.array(list(map(float, line.split()[1:])))

    def _add_stdev(self, line):
        self.stdev_angles = np.array(list(map(float, line.split()[1:])))

    def _add_rep_frame(self, line):
        cols = line.split()
        self.rep_frame = int(cols[3])
        self.rep_frame_rmsd = float(cols[4])

    def _add_circular_similarity(self, line, all_states):
        for col, state in zip(line.split()[1:], all_states):
            self.circular_similarity[state] = float(col)

    def _add_cosine_similarity(self, line, all_states):
        for col, state in zip(line.split()[1:], all_states):
            self.cosine_similarity[state] = float(col)

    def __repr__(self):
        return '<State Object: index=%d, state_code=%s>' % (
            self.index, self.state_code)


class DihedralState(object):
    def __init__(self, index, maxima, ranges):
        self.index, self.maxima, self.ranges = index, maxima, ranges


class DihedralRange(object):
    def __init__(self, min, max):
        self.min, self.max = min, max


class Dihedral(object):
    """A class representing an individually varying Dihedral angle"""
    # DihedralState = namedtuple('DihedralState', ('index', 'maxima', 'ranges'))
    # DihedralRange = namedtuple('DihedralRange', ('min', 'max'))

    def __init__(self, title, lines):
        self.index = int(re.search('[0-9]+', title).group())
        maxima = re.search('(-?[0-9]+(, )?)+', lines[0]).group().split(',')
        self.maxima = list(map(float, maxima))

        self.states = self._parse_states(lines[1], maxima)
        self.state_trajectory = []
        self.trajectory = None

    def _parse_states(self, line, maxima):
        line = line.split(':')[1]
        states = []
        state_ranges = defaultdict(lambda: [])
        for state in re.split('\),', line):
            state_index, sub_range = state.split('=')
            sub_range = sub_range.strip().strip('[)')
            sub_range = map(float, sub_range.split(','))
            state_ranges[int(state_index)].append(
                DihedralRange(*sub_range))
        for index, maximum in zip(state_ranges, maxima):
            states.append(DihedralState(index, float(maximum),
                                        state_ranges[index]))
        return states

    def state(self, angle):
        """For the provided angle return the corresponding state index
        for this dihedral"""
        for state in self.states:
            for sub_range in self.states[state]:
                if sub_range.min <= angle <= sub_range.max:
                    return state
        raise ValueError('Could not locate state for provided angle')

    def get_state(self, index=None):
        for state in self.states:
            if state.index == index:
                return state
        raise ValueError("No DihedralState with specified values: index=%s" %
                         index)


