### Machine Learning approach for automatic recognition of tomato-pollinating bees based on their buzzing-sounds

The bee-mediated pollination greatly increases the size and weight of tomato fruits. Therefore, distinguished among the local set of bees, those that are efficient pollinators are essential to improve the economic returns to farmers. To achieve these purposes, it becomes primordial to know the identity of the visiting bees. Nevertheless, the traditional taxonomic identification of bees is not an easy task, requiring the participation of experts and the use of specific equipment. Due to these limitations, the development, and implementation of new technologies for automatic recognition of bees become relevant. Based on this, we aim to verify the capacity of Machine Learning (ML) algorithms to recognize the taxonomic identity of visiting bees of tomato flowers based on the characteristics of their buzzing-sounds. We compared the performance of the ML algorithms with results from fundamental frequency analyses realized on the same data set, leading to a direct comparison of the two methods. Some classifiers, especially the SVM, achieved better performance in relation to the randomized and sound frequency-based trials. The ML classifiers presented a better performance in recognizing bee species based on their buzzing sounds over fundamental frequency. The buzzing sounds produced during sonication were more relevant for the taxonomic recognition of bee species than the flight sounds. On the other hand, the ML classifiers achieve better performance to recognize bees genera based on flight sounds. The ML techniques could lead to automate the taxonomic recognition of flower-visiting bees of the tomato crop. This would be a convenient option for professionals with no experience in bee taxonomy who are involved with the managing of tomato crops.
Future studies may focus on the technological application of this model.

#### Libs and Tools
- MFCC
- Numpy
- Pandas
- Librosa
- Sklearn

#### Baselines folder
- You can find all the baseline codes in the paper.

#### ExploringData (codes) folder
- It finds all the codes implemented to perform exploratory analysis on the data.

#### GenderCodes folder
- You will find the feature extraction code and the code with machine learning algorithms.

#### Species folder
- You will find the feature extraction code and the code with machine learning algorithms.

#### Data folder
- GenresRecordings: recordings organized by genres.
- SpeciesRecordings: recordings organized by species.
- TableRecordings: table with information about recordings.
