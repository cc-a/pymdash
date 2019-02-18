import argparse
import mdash
import mdtraj as md

parser = argparse.ArgumentParser(
    description='This script demonstrates integration of pymdash with mdtraj'
    ' to extract the representative frame for each dash state from a raw '
    'trajectory file.')
parser.add_argument(
    'dash_out_file', help='Path to the output file from dash')
parser.add_argument(
    'trajectory_file', help='Path to the trajectory data file used to provide'
    ' input for the dash run.')
parser.add_argument(
    '-t', '--topology', help='Path to a suitable structure file that can be '
    'used to derive a topology by mdtraj. See md_traj.load_toplogy for details')
args = parser.parse_args()

if args.topology is not None:
    topology = md.load_topology(args.topology)
else:
    topology = None

with open(args.dash_out_file) as f:
    dash = mdash.DashOutput(f)

print('Writing pdb files for {} dash states'.format(dash.n_states))
for state in dash.states:
    traj = md.load_frame(args.trajectory_file, state.rep_frame-1, top=topology)
    traj.save_pdb('state_{}.pdb'.format(state.index))
