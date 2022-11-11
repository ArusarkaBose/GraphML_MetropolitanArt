# GraphML_MetropolitanArt

## Feature-Based Methods
The easiest approach to conduct Supervised Learning is to use graph measures as features in a new dataset or in addition to an existing dataset.
Depending on the prediction task, we could compute node-level, edge-level, and graph-level metrics. These metrics can serve as rich information about the entity itself and also its relationship to other entities. This can be seen as a classical Supervised ML task, but the emphasis here is on Feature Selection. Depending on our prediction task, you may choose different graph metrics to predict for the label. In our dataset, the label we’ll choose is the “Highlight” field.
<br><br>
We print the value counts for **Highlight = 0** and **Highlight = 1**

![image](https://user-images.githubusercontent.com/51702854/201316427-b34af279-88aa-481c-909e-e36b67308b35.png)

We see a large class imbalance here, which makes sense since only some artworks are able to be highlighted and usually museums contain a vast collection in their archives.
