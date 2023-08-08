import os
#import dot2tex
from graphviz import Digraph
import iai_utils.decision_tree_funcs as dt_funcs


def plot(filename, tree, file_type='pdf', view=True):

    g = Digraph(
        format=file_type,

        graph_attr=dict(margin='0.01',
                        # nodesep='0.4',
                        ranksep='0.2',
                        # splines='curved'
                        ),
        engine='dot',
    )
    g.body.extend(['rankdir=TB'])


    #...writing text to nodes...
    total_nodes = tree['total_nodes']
    for i in range(total_nodes):
        my_node = dt_funcs.extract_node_from_id(tree,i)
        g.node(str(my_node['node_id']), label = my_node['node_text'],
               style='filled', fillcolor='whitesmoke', shape='rect',
               align='center', fontsize='15', height='0.3', width='0.8',
               penwidth='2', fontname="garamond")

    #..defining edges...
    for i in range(total_nodes):
        my_node = dt_funcs.extract_node_from_id(tree,i)
        my_node_id = my_node['node_id']
        if my_node['node_type'] == 'active':
            left_node_id = my_node['left_node']['node_id']
            right_node_id = my_node['right_node']['node_id']

            g.edge(str(my_node_id), str(left_node_id),
                   arrowhead="normal", penwidth='2', style='dashed', weight='10', arrowsize='.5')

            g.edge(str(my_node_id), str(right_node_id),
                   arrowhead="normal", penwidth='2', style='dashed', weight='10', arrowsize='.5')



    g.render(filename, view=view)

