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

#### PLOS Computational Biology
[PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009426)

#### Citation
```bibtex
@article{10.1371/journal.pcbi.1009426,
    doi = {10.1371/journal.pcbi.1009426},
    author = {Ribeiro, Alison Pereira AND da Silva, Nádia Felix Felipe AND Mesquita, Fernanda Neiva AND Araújo, Priscila de Cássia Souza AND Rosa, Thierson Couto AND Mesquita-Neto, José Neiva},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Machine learning approach for automatic recognition of tomato-pollinating bees based on their buzzing-sounds},
    year = {2021},
    month = {09},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pcbi.1009426},
    pages = {1-21},
    abstract = {Bee-mediated pollination greatly increases the size and weight of tomato fruits. Therefore, distinguishing between the local set of bees–those that are efficient pollinators–is essential to improve the economic returns for farmers. To achieve this, it is important to know the identity of the visiting bees. Nevertheless, the traditional taxonomic identification of bees is not an easy task, requiring the participation of experts and the use of specialized equipment. Due to these limitations, the development and implementation of new technologies for the automatic recognition of bees become relevant. Hence, we aim to verify the capacity of Machine Learning (ML) algorithms in recognizing the taxonomic identity of visiting bees to tomato flowers based on the characteristics of their buzzing sounds. We compared the performance of the ML algorithms combined with the Mel Frequency Cepstral Coefficients (MFCC) and with classifications based solely on the from fundamental frequency, leading to a direct comparison between the two approaches. In fact, some classifiers powered by the MFCC–especially the SVM–achieved better performance compared to the randomized and sound frequency-based trials. Moreover, the buzzing sounds produced during sonication were more relevant for the taxonomic recognition of bee species than analysis based on flight sounds alone. On the other hand, the ML classifiers performed better in recognizing bees genera based on flight sounds. Despite that, the maximum accuracy obtained here (73.39% by SVM) is still low compared to ML standards. Further studies analyzing larger recording samples, and applying unsupervised learning systems may yield better classification performance. Therefore, ML techniques could be used to automate the taxonomic recognition of flower-visiting bees of the cultivated tomato and other buzz-pollinated crops. This would be an interesting option for farmers and other professionals who have no experience in bee taxonomy but are interested in improving crop yields by increasing pollination.},
    number = {9},

}
´´´
