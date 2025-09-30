#ROADMAP EXTRACTION and preprocessing as a graph
#This program save public info of the road map in osm graphs format and geodataframe 
import numpy as np
import osmnx as ox
import networkx as nx


#Extract a circle map function:
def roadgraph(centre_point,radius): 
    #Extract osm graph (nodes=intersections,edges=streets) and save in variable G
    G = ox.graph_from_point(centre_point,radius, dist_type='bbox', network_type='drive', simplify=True)
    #Remove the not strongly conected components of G (otherwise diameter=inifnity)
    if not nx.is_strongly_connected(G):
        G = nx.subgraph(G, max(nx.strongly_connected_components(G), key=len)).copy()
    #Obtain the line graph (nodes=streets,edges=intersection) and save in variable line_G 
    line_G = nx.line_graph(G)
    #add labels from original graph to line_G
    line_G.add_nodes_from((node, G.edges[node]) for node in line_G)

    # add edge information
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    #Just for PLOTS nx.draw()
    #calculate the pos of edge. Using the mean of Linestring, (Linestring is the shape of the street, you can also use the median). 
    #If Linestring is not recorded, still use the mean of xy of two cross.
    pos = {}
    for node, data in line_G.nodes(data=True):
        if 'geometry' in data:
            pos[node] = np.mean(np.array(data['geometry'].coords.xy), axis=1)
        else:
            pos[node] = np.mean(np.array([[G.nodes[node[0]]['x'], G.nodes[node[1]]['x']], [G.nodes[node[0]]['y'], G.nodes[node[1]]['y']]]), axis=1)

    return (line_G,G,pos)