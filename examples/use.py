import matplotlib.pyplot as plt
import mdash

with open('out/pp00_dash.dat') as f, open('out/pp00_dash.in') as f2:
    dash = mdash.DashOutput(f, f2)

dash.reindex()

fig, ax = plt.subplots()
ax.plot([state.index for state in dash.state_trajectory], 'x')

d = 2
N = None
fig, ax = plt.subplots()
dihedral = dash.dihedrals[d]
ax.plot(dihedral.trajectory[:N])
# ax.twinx().plot([state.index for state in dash.state_trajectory[:N]], 'k')
# ax.twinx().plot(
#     [state.index for state in dihedral.state_trajectory[:N]], 'k')
ax.plot([state.mean_angles[d] for state in dash.state_trajectory[:N]], 'k')

plt.show()
