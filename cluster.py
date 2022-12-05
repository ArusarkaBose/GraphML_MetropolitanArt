from karateclub import Graph2Vec
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import AverageEmbedder, HadamardEmbedder, WeightedL1Embedder, WeightedL2Embedder
import numpy as np
import pandas as pd
from pylab import rcParams
import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


def GetGraphMetrics(graph):
    
    graph_degree = dict(graph.degree)
    print("Graph Summary:")
    print(f"Number of nodes : {len(graph.nodes)}")
    print(f"Number of edges : {len(graph.edges)}")
    print(f"Maximum degree : {np.max(list(graph_degree.values()))}")
    print(f"Minimum degree : {np.min(list(graph_degree.values()))}")
    print(f"Average degree : {np.mean(list(graph_degree.values()))}")
    print(f"Median degree : {np.median(list(graph_degree.values()))}")
    print("")
    print("Graph Connectivity")
    try:
        print(f"Connected Components : {nx.number_connected_components(graph)}")
    except:
        print(f"Strongly Connected Components : {nx.number_strongly_connected_components(graph)}")
        print(f"Weakly Connected Components : {nx.number_weakly_connected_components(graph)}")
    print("")
    print("Graph Distance")
    try:
        print(f"Average Distance : {nx.average_shortest_path_length(graph)}")
        print(f"Diameter : {nx.algorithms.distance_measures.diameter(graph)}")
    except:
        shortest_lengths = []
        for C in nx.strongly_connected_components(graph):
            shortest_lengths.append(nx.average_shortest_path_length(G.subgraph(C)))
        print(f"Average Shortest Lengths of Strongly Connected Components : {np.mean(shortest_lengths)}")
    print("")
    print("Graph Clustering")
    print(f"Transitivity : {nx.transitivity(graph)}")
    print(f"Average Clustering Coefficient : {nx.average_clustering(graph)}")
    
    return None


def CreateDataframeForCentralityMeasures(sorted_centrality_teams, centrality_name):
    df = pd.DataFrame(sorted_centrality_teams, columns=['Node', centrality_name])
    return df

if __name__=="__main__":
    metobjects = pd.read_csv('MetObjects.csv', encoding='utf8')
    metobjects_nonan = metobjects.copy()
    
    for c in metobjects_nonan.columns:
        metobjects_nonan[c] = metobjects_nonan[c].fillna(f'unknown_{c}')
    metobjects_nonan_paintings = metobjects_nonan[metobjects_nonan['Classification']=='Paintings']
    metobjects_nonan_paintings = metobjects_nonan_paintings.replace(r'\r\n','', regex=True)
    metobjects_nonan_paintings['Highlight'] = np.where(metobjects_nonan_paintings['Is Highlight']==True, 'Highlight', 'No Highlight')

    print(metobjects_nonan_paintings[['Culture', 'Period', 'Artist Display Name', 'Medium', 'Object Name', 'Title', 'Highlight']])
    metobjects_nonan_paintings.to_csv("metaobjects_nonan_paintings.csv",index=False)

    met_graph_df = metobjects_nonan_paintings[['Artist Display Name', 'Title']].rename(columns={'Artist Display Name':'From', 'Title':'To'}).append(
                    metobjects_nonan_paintings[['Artist Display Name', 'Culture']].rename(columns={'Artist Display Name':'From', 'Culture':'To'}), ignore_index=True).append(
                        metobjects_nonan_paintings[['Culture', 'Title']].rename(columns={'Culture':'From', 'Title':'To'}), ignore_index=True).append(
                            metobjects_nonan_paintings[['Artist Display Name', 'Period']].rename(columns={'Artist Display Name':'From', 'Period':'To'}), ignore_index=True).append(
                                metobjects_nonan_paintings[['Period', 'Title']].rename(columns={'Period':'From', 'Title':'To'}), ignore_index=True).append(
                                    metobjects_nonan_paintings[['Artist Display Name', 'Medium']].rename(columns={'Artist Display Name':'From', 'Medium':'To'}), ignore_index=True).append(
                                        metobjects_nonan_paintings[['Medium', 'Title']].rename(columns={'Medium':'From', 'Title':'To'}), ignore_index=True).append(
                                            metobjects_nonan_paintings[['Artist Display Name', 'Department']].rename(columns={'Artist Display Name':'From', 'Department':'To'}), ignore_index=True).append(
                                                metobjects_nonan_paintings[['Department', 'Title']].rename(columns={'Department':'From', 'Title':'To'}), ignore_index=True).append(
                                                    metobjects_nonan_paintings[['Artist Display Name', 'Object Name']].rename(columns={'Artist Display Name':'From', 'Object Name':'To'}), ignore_index=True).append(
                                                        metobjects_nonan_paintings[['Object Name', 'Title']].rename(columns={'Object Name':'From', 'Title':'To'}), ignore_index=True)

    print(met_graph_df)

    met_graph = nx.from_pandas_edgelist(met_graph_df, 'From', 'To')
    met_subgraph = met_graph.subgraph(np.array(np.random.choice(list(met_graph.nodes), int(len(list(met_graph.nodes))/2))))

    # GetGraphMetrics(met_graph)

    d = dict(met_graph.degree)
    metric_main_df = pd.DataFrame(index=list(d.keys()))
    metric_main_df.to_csv("metric_main_df.csv",index=False)

    degree_centrality_df = CreateDataframeForCentralityMeasures(sorted(nx.degree_centrality(met_graph).items(), key=lambda x:x[1], reverse=True), 'Degree Centrality')
    closeness_centrality_df = CreateDataframeForCentralityMeasures(sorted(nx.closeness_centrality(met_graph).items(), key=lambda x:x[1], reverse=True), 'Closeness Centrality')
    betweenness_centrality_df = CreateDataframeForCentralityMeasures(sorted(nx.betweenness_centrality(met_graph).items(), key=lambda x:x[1], reverse=True), 'Betweenness Centrality')
    pagerank_df = CreateDataframeForCentralityMeasures(sorted(nx.pagerank(met_graph).items(), key=lambda x:x[1], reverse=True), 'PageRank')
    hub, auth = nx.hits(met_graph)
    hubs_df = CreateDataframeForCentralityMeasures(sorted(hub.items(), key=lambda x:x[1], reverse=True), 'Hubs')
    authorities_df = CreateDataframeForCentralityMeasures(sorted(auth.items(), key=lambda x:x[1], reverse=True), 'Authorities')

    for metric in [degree_centrality_df, closeness_centrality_df, betweenness_centrality_df, pagerank_df, hubs_df, authorities_df]:
        metric = metric.set_index('Node')
        metric_main_df = metric_main_df.join(metric)
        
    metric_main_df['Log Flow Centrality'] = np.log(metric_main_df['Betweenness Centrality']+np.min(metric_main_df[metric_main_df['Betweenness Centrality']>0]['Betweenness Centrality']))
    stdscaler = StandardScaler()
    metric_main_df['Normalized Flow Centrality'] = stdscaler.fit_transform(metric_main_df['Log Flow Centrality'].values.reshape(-1,1))

    for n in met_graph.nodes():
        met_graph.nodes[n]['name'] = n
        met_graph.nodes[n]['degree'] = d[n]
        met_graph.nodes[n]['degree_centrality'] = metric_main_df[metric_main_df.index==n]['Degree Centrality'].values[0]
        met_graph.nodes[n]['closeness_centrality'] = metric_main_df[metric_main_df.index==n]['Closeness Centrality'].values[0]
        met_graph.nodes[n]['betweenness_centrality'] = metric_main_df[metric_main_df.index==n]['Betweenness Centrality'].values[0]
        met_graph.nodes[n]['flow_centrality'] = metric_main_df[metric_main_df.index==n]['Normalized Flow Centrality'].values[0]
        met_graph.nodes[n]['pagerank'] = metric_main_df[metric_main_df.index==n]['PageRank'].values[0]
        met_graph.nodes[n]['hubs'] = metric_main_df[metric_main_df.index==n]['Hubs'].values[0]
        met_graph.nodes[n]['authorities'] = metric_main_df[metric_main_df.index==n]['Authorities'].values[0]
        
    for e in met_graph.edges():
        met_graph.edges[(e[0],e[1])]['jac'] = [p for u,v,p in nx.jaccard_coefficient(met_graph, [(e[0], e[1])])][0]
        met_graph.edges[(e[0],e[1])]['rai'] = [p for u,v,p in nx.resource_allocation_index(met_graph, [(e[0], e[1])])][0]
        met_graph.edges[(e[0],e[1])]['pa'] = [p for u,v,p in nx.preferential_attachment(met_graph, [(e[0], e[1])])][0]
        
    pos = nx.kamada_kawai_layout(met_graph)

    options = {'node_size': 50,
            'linewidths': .2,
            'width': 0.25}

    random.seed(0)
    node2vec = Node2Vec(met_graph, dimensions=20, walk_length=10, num_walks=100)
    model = node2vec.fit(window=10)
    embeddings = model.wv

    fig, ax = plt.subplots(figsize=(25,15))

    for x in met_graph.nodes():
        if (x in list(metobjects_nonan_paintings.groupby('Artist Display Name').size().sort_values(ascending=False)[:50].index)) & (x not in ['unknown_Artist Display Name', 'Unidentified Artist']):
        
            v = model.wv[str(x)]
            ax.scatter(v[0],v[1], s=1000)
            ax.annotate(str(x), (v[0],v[1]), fontsize=16)

    plt.show()

    #Artists similar to Vincent van Gogh
    for artist,_ in model.wv.most_similar('Vincent van Gogh'):
        if (artist in list(metobjects_nonan_paintings['Artist Display Name'].unique())) & (artist not in ['unknown_Artist Display Name', 'Unidentified Artist']):
            print(artist)

    edges_embs_avg = AverageEmbedder(keyed_vectors=model.wv)
    edges_embs_had = HadamardEmbedder(keyed_vectors=model.wv)
    edges_embs_wl1 = WeightedL1Embedder(keyed_vectors=model.wv)
    edges_embs_wl2 = WeightedL2Embedder(keyed_vectors=model.wv)

    vangogh_similar = ['Vincent van Gogh', 'Paul Cézanne', 'Goya (Francisco de Goya y Lucientes)', 'Georges Rouault', 'Alfred Sisley', 'Georges Seurat', 'Claude Monet', 'Gustave Courbet', 'Edward McKnight Kauffer']
    fig, ax = plt.subplots(figsize=(25,15))

    for x in met_graph.edges():
        if (x[0] in vangogh_similar) & (x[1] in list(metobjects_nonan_paintings['Title'].unique())):
            
            v = edges_embs_avg[(str(x[0]), str(x[1]))]
            ax.scatter(v[0],v[1], s=1000)
            ax.annotate(str(x), (v[0],v[1]), fontsize=16)

    plt.show()

    vangogh_similar_graphs = []

    for artist_similar_to_vangogh in vangogh_similar:
        vangogh_similar_paintings = metobjects_nonan_paintings[metobjects_nonan_paintings['Artist Display Name']==artist_similar_to_vangogh]
        artist_subgraph_df = vangogh_similar_paintings[['Artist Display Name', 'Title']].rename(columns={'Artist Display Name':'From', 'Title':'To'}).append(
                                vangogh_similar_paintings[['Artist Display Name', 'Culture']].rename(columns={'Artist Display Name':'From', 'Culture':'To'}), ignore_index=True).append(
                                    vangogh_similar_paintings[['Culture', 'Title']].rename(columns={'Culture':'From', 'Title':'To'}), ignore_index=True).append(
                                        vangogh_similar_paintings[['Artist Display Name', 'Period']].rename(columns={'Artist Display Name':'From', 'Period':'To'}), ignore_index=True).append(
                                            vangogh_similar_paintings[['Period', 'Title']].rename(columns={'Period':'From', 'Title':'To'}), ignore_index=True).append(
                                                vangogh_similar_paintings[['Artist Display Name', 'Medium']].rename(columns={'Artist Display Name':'From', 'Medium':'To'}), ignore_index=True).append(
                                                    vangogh_similar_paintings[['Medium', 'Title']].rename(columns={'Medium':'From', 'Title':'To'}), ignore_index=True).append(
                                                        vangogh_similar_paintings[['Artist Display Name', 'Department']].rename(columns={'Artist Display Name':'From', 'Department':'To'}), ignore_index=True).append(
                                                            vangogh_similar_paintings[['Department', 'Title']].rename(columns={'Department':'From', 'Title':'To'}), ignore_index=True).append(
                                                                vangogh_similar_paintings[['Artist Display Name', 'Object Name']].rename(columns={'Artist Display Name':'From', 'Object Name':'To'}), ignore_index=True).append(
                                                                    vangogh_similar_paintings[['Object Name', 'Title']].rename(columns={'Object Name':'From', 'Title':'To'}), ignore_index=True)
        artist_subgraph = nx.from_pandas_edgelist(artist_subgraph_df, 'From', 'To')
        vangogh_similar_graphs.append(artist_subgraph)
        
    graphs_model = Graph2Vec(dimensions=2)
    graphs_model.fit([nx.convert_node_labels_to_integers(x) for x in vangogh_similar_graphs])
    graph_embeddings = graphs_model.get_embedding()

    fig, ax = plt.subplots(figsize=(25,10))

    for i,vec in enumerate(graph_embeddings):
        ax.scatter(vec[0],vec[1], s=1000)
        ax.annotate(str(vangogh_similar[i]), (vec[0],vec[1]), fontsize=16)

    train_test_df = metobjects_nonan_paintings[['Title', 'Highlight']].set_index('Title').join(metric_main_df, how='left')
    train_test_df['Highlight'] = np.where(train_test_df['Highlight']=='Highlight', 1, 0)
    train_test_df['Highlight'].value_counts()

    print(train_test_df.head())

    X = train_test_df.drop(['Highlight'], axis=1)
    y = train_test_df['Highlight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ros = RandomOverSampler(random_state=0)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=1))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=100, verbose=2, callbacks=[callback], shuffle=False)
    y_pred = model.predict(X_test)

    print('Accuracy', accuracy_score(y_test, y_pred))
    print('Precision', precision_score(y_test, y_pred))
    print('Recall', recall_score(y_test, y_pred))
    print('F1-score', f1_score(y_test, y_pred))
