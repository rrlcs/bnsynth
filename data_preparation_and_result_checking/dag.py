import networkx as nx
import matplotlib.pyplot as plt

def resolve_dependency():
	f = open("dag.txt", "r")
	dag_data = f.read()
	nodes = dag_data.split("\n\n")[0].split("\n")
	edges_with_labels = dag_data.split("\n\n")[1].split("\n")
	edges = [ed.split(",") for ed in edges_with_labels]
	G = nx.DiGraph()
	G.add_nodes_from(nodes)
	for i in range(len(edges)):
		G.add_edge(edges[i][0], edges[i][1], eqn=edges[i][2])
	nx.draw(G, with_labels=True, node_size=700)
	plt.savefig("graph.png")
	plt.show()
	plt.close()

	reverse_topological_sort = list(reversed(list(nx.topological_sort(G))))
	return reverse_topological_sort