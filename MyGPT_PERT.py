#MyGPT_PERT

"""
Calculating the critical path in project management involves determining the longest path through a project network, identifying tasks that cannot be delayed without delaying the entire project. Python doesn't have a built-in library for project management, but you can use network analysis libraries like networkx for this purpose.

Here's an example program using networkx to calculate the critical path:

python
Copy code

"""

import networkx as nx
import matplotlib.pyplot as plt

def calculate_critical_path(graph):
    # Calculate earliest start and finish times
    es = nx.get_node_attributes(graph, 'es')
    for node in graph.nodes():
        es[node] = max([es[predecessor] + graph[predecessor][node]['duration'] for predecessor in graph.predecessors(node)], default=0)

    # Calculate latest start and finish times
    ls = nx.get_node_attributes(graph, 'ls')
    for node in list(reversed(list(graph.nodes()))):
        ls[node] = min([ls[successor] - graph[node][successor]['duration'] for successor in graph.successors(node)], default=es[node])

    # Calculate slack times
    slack_times = {node: ls[node] - es[node] for node in graph.nodes()}

    # Identify critical path
    critical_path = [node for node, slack in slack_times.items() if slack == 0]

    return critical_path, slack_times

# Example project network
project_graph = nx.DiGraph()

# Define tasks and durations
tasks = {
    'Start': 0,
    'A': 5,
    'B': 3,
    'C': 2,
    'D': 7,
    'E': 4,
    'F': 6,
    'End': 0,
}

# Add nodes and edges to the graph
for task, duration in tasks.items():
    project_graph.add_node(task, duration=duration)

project_graph.add_edges_from([('Start', 'A'), ('Start', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('C', 'E'), ('D', 'F'), ('E', 'F'), ('F', 'End')])

# Set earliest start time for the Start node
nx.set_node_attributes(project_graph, {node: 0 for node in project_graph.nodes()}, 'es')

# Calculate critical path and slack times
critical_path, slack_times = calculate_critical_path(project_graph)

# Print results
print("Critical Path:", critical_path)
print("Slack Times:", slack_times)

# Draw the project network
pos = nx.spring_layout(project_graph)
nx.draw(project_graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, font_color='black')
edge_labels = {(i, j): project_graph[i][j]['duration'] for i, j in project_graph.edges()}
nx.draw_networkx_edge_labels(project_graph, pos, edge_labels=edge_labels, font_color='red')

plt.title('Project Network')
plt.show()

"""
In this example:

The project network is represented as a directed acyclic graph (DAG) using networkx.
Node attributes represent the duration of each task.
The program calculates the earliest start and finish times, latest start and finish times, slack times, and identifies the critical path.
The project network is visualized using matplotlib.
Make sure to install networkx before running the script:

bash
Copy code
pip install networkx
Adjust the tasks and durations in the tasks dictionary based on your project requirements.

"""
