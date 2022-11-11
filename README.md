# GraphML_MetropolitanArt

### Node2Vec
The way the random walks are created in the Node2Vec Deep Walk model differs noticeably from that of the Deep Walk model. Because Deep Walk is limited in its ability to preserve a node's local neighbourhood information, Node2Vec does the graph exploration using a flexible and effective combination of Breadth-First Search (BFS) and Depth-First Search (DFS). We do so by:
1. p: the likelihood that a random stroll will return to the first node.
2. q: the likelihood that a random walk will go across a portion of the graph that hasn't been seen before. BFS/DFS ration is the ratio.

![image](https://user-images.githubusercontent.com/42794447/201329387-75b0ee01-3f67-4628-89b3-8848b0bdca61.png)

We see collection of artists being generated calculated via the generated walks. We added Medium, Department, and Object Type in the graph. So, some artists differing in the Period but using common materials appear in the same Department of the Museum. The search results for Vincent van Gogh are:

![image](https://user-images.githubusercontent.com/42794447/201330066-0699ad3d-6a41-4d62-af60-d10eb974bd0c.png)

### Edge2Vec
Edge2Vec is a fairly straightforward version of Node2Vec in which the embedding of the edge linking two neighbouring nodes is extracted using some elementary mathematical operations on the node embedding. Depending on the nature of your data and how it may relate to each other, different embeddings will be of value. An effective evaluation method can be to try each of them and see which has the most sensible separation of edges.

![image](https://user-images.githubusercontent.com/42794447/201333794-d168070f-9d2f-4b28-a5f0-abe34e79f449.png)

The image outlines collection of artists being generated calculated via the generated walks.

### Graph2Vec
The learning of node and edge representations is most generalised in this way. This model is known as Doc2Vec in natural language processing since it embeds numerous documents rather than single words or phrases. Graphs can be broken down into smaller units called subgraphs, and the model can be trained to learn an embedding that represents each of these subgraphs.

![image](https://user-images.githubusercontent.com/42794447/201334165-40e96855-5f49-479a-a7ac-3b3e8dd9a82e.png)

Simplified graphical representation of the Doc2Vec skip-gram model. The number of d neurons in the hidden layer represents the final size of the embedding space. We can have the following three different ways subgraphs may be created.

1. Embed nodes and aggregate them (sum or average most common).
2. Create a node (super-node) that symbolizes and spans each subgraph and then embed that node.
3. Anonymous Walk Embeddings. Capture states that correspond to the index of the first time a node is visited in a random walk. Considered anonymous because this method is agnostic to the identity of the nodes visited.

![image](https://user-images.githubusercontent.com/42794447/201336600-10ff219a-ae05-4dd4-906b-2b433034f299.png)

To express the graph as a probability distribution over these walks is the main objective.
In our situation, we can make subgraphs using just the rows that contain works by Vincent van Gogh and the other artists the model identified as being related to him. We can express all of their data using embeddings in a far more compact and effective manner. Note that for the model to fit properly, the node labels must be converted to integers. The next figure shows the results for the same search for Graph2Vec.
![image](https://user-images.githubusercontent.com/42794447/201334922-ad868e4e-46c4-49fd-81ba-3fde163e8467.png)

This shows how Without target labels or ground truth values, we can use graphs to find underlying structural similarities really effectively and efficiently.

## Feature-Based Methods
The easiest approach to conduct Supervised Learning is to use graph measures as features in a new dataset or in addition to an existing dataset.
Depending on the prediction task, we could compute node-level, edge-level, and graph-level metrics. These metrics can serve as rich information about the entity itself and also its relationship to other entities. This can be seen as a classical Supervised ML task, but the emphasis here is on Feature Selection. Depending on our prediction task, you may choose different graph metrics to predict for the label. In our dataset, the label we’ll choose is the “Highlight” field.
<br><br>
We print the value counts for **Highlight = 0** and **Highlight = 1**

![image](https://user-images.githubusercontent.com/51702854/201316427-b34af279-88aa-481c-909e-e36b67308b35.png)

We see a large class imbalance here, which makes sense since only some artworks are able to be highlighted and usually museums contain a vast collection in their archives.

On looking at the dataset, we see a variety of features that can be used to predict the **Highlight** value.

![image](https://user-images.githubusercontent.com/51702854/201317108-56d1783e-9cdc-474e-9294-7dcc055f608d.png)

We randomly oversample our dataset and then predict for the target variable.

![image](https://user-images.githubusercontent.com/51702854/201317464-f5f8a468-923b-4b74-b655-e3d4a379bbdb.png)

Though the results are not extremely impressive, it is interesting to note that we just used the base model and thus better performance can be obtained with extensive tuning or feature engineering. The reason we optimize for recall in our problem is because it’s better to get “relevant” results than “exact” results in this case, so it is good to see a relatively high recall score.
<br><br>
Finally, this was a node-level prediction task where the node in question is an artwork. This same concept can really easily be done for edge or graph-level (with traditional features) tasks as well making it highly versatile.
