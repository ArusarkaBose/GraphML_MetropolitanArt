# Artwork Clustering and Its Applications

## Introduction
In this project, we aim to draw conclusions and relationships between a dataset using Graph Machine Learning. Representing data is represented as a graph allows us to embed the rich structural information as features. This results in better performance in some areas since relational structures can offer a wealth of useful data. The task is to identify relationships between existing artworks in the dataset and cluster the artworks into categories. We want to develop clusters that identify artworks that are similar in style and content. This finds implementation in better-structured artwork, understanding the role of time and space in the history of art and several unique art forms available and how they relate to each other.

## Brief Outline of the Dataset
The Metropolitan Museum of Art presents over 5,000 years of art from around the world for everyone to experience and enjoy. The Museum lives in two iconic sites in New York City—The Met Fifth Avenue and The Met Cloisters. Millions of people also take part in The Met experience online. The Metropolitan Museum of Art provides select datasets of information on more than 470,000 artworks in its Collection for unrestricted commercial and noncommercial use. The dataset is available in CSV format, encoded in UTF-8. 

**Link of the dataset** - https://github.com/metmuseum  [.CSV file available under open access]

**Dataset Summary** - The dataset has categories like 'Culture', 'Period', 'Artist Display Name', 'Medium', 'Object Name', ‘Dynasty’ and ‘Domain’. These categories help us identify several classes and clusters. We plan on using several graph structure based features like centrality to identify the importance of all of them to design the graph and apply the required algorithms to produce classification classes.

## Reason behind this project
Each and every artist has a distinctive style that differentiates his/her painting from the rest and often it is the muse of art connoisseurs to ascribe a painting to someone. This project will aim to develop a model which can be used to classify or cluster paintings belonging to a single artist or multiple artists with the same style of work. The work can be further extended to ferret out fakes from original works in cases where it becomes difficult even for a trained eye. 

Not just that, similar algorithms and work seem to be applicable in every domain that will flourish with user/product clustering - like e-commerce, business marketing, and identifying the target audience. 

## Data Visualization
At the start of the project, we explore data and conduct some exploratory data analysis to visualize the dataset. With the help of network topology visualisation, we can understand what kind of data we are dealing with. The dataset consists of over 450K rows which would take a very long time to encode in a graph. Hence, we filter down to reduce the complexity of the graph. 

**Segregration based on Department** - We found that other than country and artist, another key feature useful for clustering artworks is the department of the art. Arts from the Medieval Period seemed extremely close to each other. We saw their distribution as well and plot the department if the count is greater than 1000.

![newplot (2)](https://user-images.githubusercontent.com/42794447/205672094-bfeb0202-4f92-4a39-b53a-dcd08c7cb34d.png)

**Identification of Relevant Artists** - Next, we aimed to identifying the distribution of different artists and their artwork. It goes without saying that some paintings from an artist will be extremely related and should be identified as a part of a similar group while the same might not be true for some of the paintings of the same artist. Also, it might be related to different paintings by a different artist. 

![newplot (1)](https://user-images.githubusercontent.com/42794447/205666023-dabd7bd4-1e8a-4935-87ea-ac6be6d21b03.png)

**Identification of Relevant Countries** - We start the analysis by looking at the countries that provided different pictures and what countries were more relevant in the past in the art work domain. We only considered countries that atleast provided 100 images. We also observed from the distribtuion that only nine of the countries have a contribution of two percent and more than 80% of the total distribution. So, we clearly see the country of the artwork is an important aspect for identifying relevant artwork.

![newplot](https://user-images.githubusercontent.com/42794447/205661959-6bfde109-0791-4c5b-8e58-a78014e3044d.png)

**Distribution of Object Names of Artworks** -The style of the artwork is based upon the obkect type of the artwork as well. A painting is a little diffeent in style than a picture or a negative. However, the content of the image is not encoded by this feature but it gives good information about the style of the image.

![image](https://user-images.githubusercontent.com/42794447/205674064-af05a8b6-de13-4357-bc04-573885439355.png)

We observed that for a lot of features, there are a lot of values that are unknown or hard to accurately figure out. Similar observations are true for other columns as well. For now, we do not drop the unknown values from the dataset, as we would lose information about some other columns. 

## Aim
![image](https://user-images.githubusercontent.com/42794447/205700404-7acabf9c-28d8-472f-a15f-bd7d4b072f88.png)


## Graph Formation
We form an undirected graph, by creating a edge between two columns if the columns occur together. We do that by accumulating all rows for pairs of columns into a dataframe, and renaming the columns in a "From" - "To" format.

![Untitled Diagram drawio (5)](https://user-images.githubusercontent.com/42794447/205692970-6d450b1e-d72e-48d6-b8e0-b04664df4ace.png)

The new dataframe is then converted into a graph by adding an edge between the values in the "From" column to "To" column using the networkx library. To better understand the graph, we do some basic analysis on the graph. After performing some basic operations on the graph, we have the final graph results.

**Edge and Node Details**

- Number of nodes : 12688
- Number of edges : 66693

**Degree Details**

- Maximum degree : 9092
- Average degree : 10.51
- Median degree : 6


**Distance Metrics**
- Total Components - 1
- Diameter : 6
- Average Distance : 2.5

**Structure Components**
- Transitivity : 0.00153
- Average Clustering Coefficient : 0.52

We observe a very low transitivity, as this is not a social network and there is no reason to expect triadic closure to hold. In addition, we shouldn't have many subgraphs that are only connected and nothing else, therefore only 1 connected component is an interesting measure that can make sense. The amount of nodes and edges suggests that the graph is being further segmented by a large number of specialist categories, artists, genres, etc.

We calculate various centrality measures for the graph, such as degree centrality, betweenness centrality & closeness centrality. We observe that there are a handful of nodes strongly skewing the distributions, namely the nodes with "unknown" values. When we sort by the betweenness centrality, we find that a lot of the “unknown” or NaN fields dominate.

![1_noQuWBLvzHEsOeLbtZ0FBw copy](https://user-images.githubusercontent.com/45456921/201384016-1020c08a-15fb-4848-825f-7c0405378dea.png)


## Unsupervised Embedding Models

### Node2Vec
Node2Vec is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. It learns low-dimensional representations for nodes in a graph by optimizing a neighborhood preserving objective. Node2Vec does the graph exploration using a flexible and effective combination of Breadth-First Search (BFS) and Depth-First Search (DFS). We do so by:
1. p: the likelihood that a random stroll will return to the first node.
2. q: the likelihood that a random walk will go across a portion of the graph that hasn't been seen before. BFS/DFS ration is the ratio.

![image](https://user-images.githubusercontent.com/42794447/201329387-75b0ee01-3f67-4628-89b3-8848b0bdca61.png)

We see collection of artists being generated calculated via the generated walks. We added Medium, Department, and Object Type in the graph. So, some artists differing in the Period but using common materials appear in the same Department of the Museum. The search results for Vincent van Gogh are:
- Alfred Sisley(Same Period)
- Paul Gauguin(Same Country)
- John Russell(Same Color Theory - captured)
- Gustave Courbet(Similar Works)

### Edge2Vec
In Edge2Vec, the embedding of the edge linking two neighbouring nodes is extracted using some elementary mathematical operations on the node embedding. Edge2vec takes both the local and the global structure information of edges into consideration to preserve structure information of embedded edges as much as possible. To achieve this goal, edge2vec first ingeniously combines the deep autoencoder and Skip-gram model through a well-designed deep neural network. The experimental results on different datasets show edge2vec benefits from the direct mapping in preserving the structure information of edges.

![image](https://user-images.githubusercontent.com/42794447/201333794-d168070f-9d2f-4b28-a5f0-abe34e79f449.png)

The image outlines collection of artists being generated calculated via the generated walks.

### Graph2Vec
Graph kernels use handcrafted features (e.g., shortest paths, graphlets, etc.) and hence are hampered by problems such as poor generalization. To address this limitation, we use a neural embedding framework named graph2vec to learn data-driven distributed representations of arbitrary sized graphs. graph2vec’s embeddings are learnt in an unsupervised manner and are task agnostic. Hence, they could be used for any downstream task such as graph classification, clustering and even seeding supervised representation learning approaches. The learning of node and edge representations is most generalised in this way. Graphs can be broken down into smaller units called subgraphs, and the model can be trained to learn an embedding that represents each of these subgraphs.

![image](https://user-images.githubusercontent.com/42794447/205704025-af2bae1d-cec6-44d0-8fdc-666c5385ab28.png)

Simplified graphical representation of the Doc2Vec skip-gram model. The number of d neurons in the hidden layer represents the final size of the embedding space. We can have the following three different ways subgraphs may be created.

1. Embed nodes and aggregate them (sum or average most common).
2. Create a node (super-node) that symbolizes and spans each subgraph and then embed that node.
3. Anonymous Walk Embeddings. Capture states that correspond to the index of the first time a node is visited in a random walk. Considered anonymous because this method is agnostic to the identity of the nodes visited.

![image](https://user-images.githubusercontent.com/42794447/205703689-9d257abc-6bbe-4910-a98b-809f2d5bc955.png)


To express the graph as a probability distribution over these walks is the main objective.
In our situation, we can make subgraphs using just the rows that contain works by Vincent van Gogh and the other artists the model identified as being related to him. We can express all of their data using embeddings in a far more compact and effective manner. Note that for the model to fit properly, the node labels must be converted to integers. The next figure shows the results for the same search for Graph2Vec.
![image](https://user-images.githubusercontent.com/42794447/201334922-ad868e4e-46c4-49fd-81ba-3fde163e8467.png)

This shows how without target labels or ground truth values, we can use graphs to find underlying structural similarities really effectively and efficiently.

## Training and Testing the Results
The easiest approach to conduct Supervised Learning is to use graph measures as features in a new dataset or in addition to an existing dataset.
Depending on the prediction task, we could compute node-level, edge-level, and graph-level metrics. These metrics can serve as rich information about the entity itself and also its relationship to other entities. This can be seen as a classical Supervised ML task, but the emphasis here is on Feature Selection. Depending on our prediction task, you may choose different graph metrics to predict for the label. In our dataset, the label we’ll choose is the “Highlight” field.
<br><br>
We print the value counts for **Highlight = 0** and **Highlight = 1**

![image](https://user-images.githubusercontent.com/51702854/205676051-e2d2a787-7b70-4c64-965a-7bbf2393d144.png)

We see a large class imbalance here, which makes sense since only some artworks are able to be highlighted and usually museums contain a vast collection in their archives. We also plot the *Highlight* feature.

![image](https://user-images.githubusercontent.com/51702854/205676328-bfff10e1-4c9a-4e1f-a2bb-a98566b47ba9.png)


On looking at the dataset, we see a variety of features that can be used to predict the **Highlight** value.

![image](https://user-images.githubusercontent.com/51702854/201317108-56d1783e-9cdc-474e-9294-7dcc055f608d.png)

We randomly oversample our dataset and then predict for the target variable. We use a Keras neural model with four dense layers and compile it with the ADAM optimizer. We optimize for categorical crossentropy.

![image](https://user-images.githubusercontent.com/51702854/205681577-25cc741c-471a-443a-bea1-0f33e9e660dc.png)

![image](https://user-images.githubusercontent.com/51702854/205682342-f2d9e8a7-7faa-44aa-985e-59708e08bba3.png)

Though the results are not extremely impressive, it is interesting to note that we just used the base model and thus better performance can be obtained with extensive tuning or feature engineering. The reason we optimize for recall in our problem is because it’s better to get “relevant” results than “exact” results in this case, so it is good to see a relatively high recall score.
<br><br>
Finally, this was a node-level prediction task where the node in question is an artwork. This same concept can really easily be done for edge or graph-level (with traditional features) tasks as well making it highly versatile.

## References Used
- Dataset - https://github.com/metmuseum
- Word2Vec Algorithm - https://code.google.com/archive/p/word2vec/
- Node2Vec Algorithm - https://snap.stanford.edu/node2vec/
- Edge2Vec Algorithm - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2914-2
- Graph2Vec Algorithm - https://arxiv.org/abs/1707.05005
- Claudio Stamile, Aldo Marzullo, Enrico Deusebio - Graph Machine Learning
- Medium Blog - https://towardsdatascience.com/graph-machine-learning-with-python-part-3-unsupervised-learning-aa2854fe0ff2
