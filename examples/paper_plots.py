"""This script demonstrates how to recreate some of the plots from the
original dash paper using pymdash.

David W. Salt, Brian D. Hudson, et. al., Journal of Medicinal Chemistry 2005 48 (9), 3214-3220 """

import matplotlib.pyplot as plt
import mdash

with open('data/tzd.out') as f, open('data/tzd.in') as f2:
    dash = mdash.DashOutput(f, f2)

# Figure 6
fig, ax = plt.subplots()
ax.plot([state.index for state in dash.state_trajectory], 'x')
ax.set_xlabel('Trajectory Frame')
ax.set_ylabel('Dash State Index')

# Figure 3
N = 5000
dihedral = dash.dihedrals[2]  # pick a nice example
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
ax1.plot(dihedral.trajectory[:N], 'o')
ax1.plot([state.maximum for state in dihedral.state_trajectory[:N]], 'k')
ax1.set_xlabel('Trajectory Frame')
ax1.set_ylabel('Dihedral 3 Value')

ax2.hist(dihedral.trajectory[:N], bins=50, orientation='horizontal')
ax2.invert_xaxis()
ax2.set_yticks([])

plt.show()
