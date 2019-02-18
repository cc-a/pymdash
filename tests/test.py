import mdash
import unittest
import numpy as np


class Test(unittest.TestCase):
    def test(self):
        with open('data/simple_dash.out') as f, \
             open('data/simple_dash.in') as f2:
            dash = mdash.DashOutput(f, f2)

        self.assertEqual(dash.options['data'], 'angles')
        self.assertEqual(dash.options['timestep'], '1 ps')
        self.assertEqual(dash.options['window'], 11)
        self.assertEqual(dash.options['binsize'], 4)
        self.assertEqual(dash.options['runlen'], 3)
        self.assertEqual(dash.options['fmax'], 2.4)
        self.assertEqual(dash.options['smin'], 48)
        self.assertEqual(dash.options['boutlen'], 20)
        self.assertEqual(dash.options['smooth'], 40)
        self.assertEqual(dash.options['rough'], 20)
        self.assertEqual(dash.options['repex_fraction'], 0.01)

        self.assertEqual(dash.n_states, 2)
        self.assertEqual(dash.n_dihedrals, 12)

        self.assertEqual(dash.states[0].state_code, '111112111111')
        self.assertEqual(dash.states[1].state_code, '111111111112')

        self.assertEqual(dash.states[0].rep_frame, 1)
        self.assertEqual(dash.states[1].rep_frame, 5)
        self.assertEqual(dash.states[0].rep_frame_rmsd, 6.83)
        self.assertEqual(dash.states[1].rep_frame_rmsd, 7.77)

        self.assertEqual(dash.state_trajectory[0], dash.states[0])
        self.assertEqual(dash.state_trajectory[1], dash.states[0])
        self.assertEqual(dash.state_trajectory[2], dash.states[0])
        self.assertEqual(dash.state_trajectory[3], dash.states[1])
        self.assertEqual(dash.state_trajectory[4], dash.states[1])
        self.assertEqual(dash.state_trajectory[5], dash.states[0])

        self.assertEqual(
            "".join(["%8.2f" % ang for ang in dash.states[0].mean_angles]),
            " -159.54 -161.05 -161.17   56.74 -100.26  -86.42  123.95   69.04  169.30  171.69  110.46  -17.15"
        )
        self.assertEqual(
            "".join(["%8.2f" % ang for ang in dash.states[1].mean_angles]),
            " -164.29 -153.69 -148.00   65.16 -132.78 -144.98  131.95   62.21  155.42  165.33  132.84   65.27"
        )

        self.assertEqual(
            "".join(["%8.2f" % ang for ang in dash.states[0].stdev_angles]),
            "   12.10    6.88    8.04    5.54   12.80    9.08    6.96    9.40    4.90    2.89    4.94   11.07"
        )
        self.assertEqual(
            "".join(["%8.2f" % ang for ang in dash.states[1].stdev_angles]),
            "    7.26    0.68    7.34    2.95    0.22   17.79    0.28    6.63    9.12    4.17    4.76   11.51"
        )

        dihedral = dash.dihedrals[0]
        self.assertEqual(dihedral.get_state(index=1), dihedral.states[0])

        self.assertTupleEqual(
            tuple(dihedral.trajectory),
            (-168.895278931, -173.250335693, -152.416381836, -171.548736572, -157.039138794, -143.536956787)
        )

        for i, dihedral in enumerate(dash.dihedrals):
            self.assertEqual(
                dihedral.state_trajectory,
                [dihedral.get_state(index=int(state.state_code[i]))
                 for state in dash.state_trajectory])

    def test_tzd(self):
            with open('data/tzd.out') as f, \
                 open('data/tzd.in') as f2:
                dash = mdash.DashOutput(f, f2)

            # check that the mamixum frequency value of each dihedral
            # state falls within the appropriate range of angles
            for dih in dash.dihedrals:
                for state in dih.states:
                    self.assertTrue(any(r.min < state.maximum < r.max
                                        for r in state.ranges))

    def test_reindex(self):
        dash = self.get_simple_dash()
        state1, state2, state3, state4 = dash.states
        # check reordering for simple case with no repititions
        dash.state_trajectory = [state4, state3, state2, state1]
        dash.reindex_by_occurence()
        self.assertEqual(
            [state4.index, state3.index, state2.index, state1.index],
            [1, 2, 3, 4])

        # more complex case with repititions and NullStates
        dash.state_trajectory = [state1,
                                 dash.NullState,
                                 state1,
                                 state2,
                                 state1,
                                 state3,
                                 dash.NullState,
                                 state2,
                                 state1,
                                 state4,
                                 dash.NullState,
                                 dash.NullState,
                                 state1,
                                 state3,
                                 state4]
        dash.reindex_by_occurence()
        self.assertEqual(
            [state1.index, state2.index, state3.index, state4.index],
            [1, 2, 3, 4])

    def get_simple_dash(self):
        dash = mdash.DashOutput([])
        dih1 = mdash.Dihedral(
            '1', [' : 0, 0', ': 1 = [-180, 0), 2 = [0, 180)'])
        dih2 = mdash.Dihedral(
            '2', [' : 0, 0', ': 1 = [-180, 0), 2 = [0, 180)'])
        dash.dihedrals = [dih1, dih2]
        state1 = mdash.State('1 1 1', [dih1, dih2])
        state2 = mdash.State('2 2 1', [dih1, dih2])
        state3 = mdash.State('3 1 2', [dih1, dih2])
        state4 = mdash.State('4 2 2', [dih1, dih2])
        dash.states = [state1, state2, state3, state4]
        dash._create_null_state()
        return dash

    def test_combine(self):
        dash = self.get_simple_dash()
        s1, s2, s3, s4 = dash.states
        dash.state_trajectory = [s1, s2, s3, s4]
        dash.combine_states((1, 2))
        self.assertEqual(dash.state_trajectory, [s1, s1, s3, s4])
        self.assertEqual([state.index for state in dash.state_trajectory],
                         [1, 1, 2, 3])
        self.assertEqual(dash.n_states, 3)

        dash = self.get_simple_dash()
        s1, s2, s3, s4 = dash.states
        dash.state_trajectory = [s1, s2, s3, s4]
        dash.combine_states((1, 2, 4))
        self.assertEqual(dash.state_trajectory, [s1, s1, s3, s1])
        self.assertEqual([state.index for state in dash.state_trajectory],
                         [1, 1, 2, 1])
        self.assertEqual(dash.n_states, 2)

    def test_combine_by_similarity(self):
        dash = self.get_simple_dash()
        s1, s2, s3, s4 = dash.states
        dash.state_trajectory = [s1, s2, s3, s4]

        s1.cosine_similarity = {s2: 1., s3: 0., s4: 0.}
        s2.cosine_similarity = {s1: 1., s3: 0., s4: 0.}
        s3.cosine_similarity = {s1: 0., s2: 0., s4: 1.}
        s4.cosine_similarity = {s1: 0., s2: 0., s3: 1.}

        dash.combine_states_by_similarity(0.9)
        self.assertEqual(dash.state_trajectory, [s1, s1, s3, s3])
        self.assertEqual([state.index for state in dash.state_trajectory],
                         [1, 1, 2, 2])
