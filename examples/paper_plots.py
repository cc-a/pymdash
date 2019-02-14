import matplotlib.pyplot as plt
import mdash

with open('data/tzd.out') as f, open('data/tzd.in') as f2:
    dash = mdash.DashOutput(f, f2)

fig, ax = plt.subplots()
ax.plot([state.index for state in dash.state_trajectory], 'x')

d = 5  # index of dihedral to plot
N = None
fig, ax = plt.subplots()
dihedral = dash.dihedrals[d]
ax.plot(dihedral.trajectory[:N])
ax.plot([state.mean_angles[d] for state in dash.state_trajectory[:N]], 'k')

fig, ax = plt.subplots()
ax.plot(dihedral.trajectory[:N])
twin_ax = ax.twinx()
twin_ax.plot(
    [state.index for state in dihedral.state_trajectory[:N]], 'k')


plt.show()
