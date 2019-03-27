# Transductive Classification of Chinese Hypernymy vs. Non-hypernymy Relations Based on Word Emebeddings

### By Chengyu Wang (https://chywang.github.io)

**Introducion:** This software classifies Chinese word pairs into hypernymy vs. non-hypernymy relations based on transductive non-linear projection learning. Two datasets of Chinese word pairs (i.e., a training set and a testing set), together with all the embedding vectors of associated Chinese word pairs should be provided as inputs. The software authormantically trains the model and makes predictions over the testing set.

**Paper:** Wang et al. Transductive Non-linear Learning for Chinese Hypernym Prediction. ACL 2017


**APIs**

+ TransductLeaner: The main software entry-point, with five input argements required.

1. w2vPath: The embeddings of all Chinese words in either the training set or the testing set. The start of each line of the file is the Chinose word, followed by the embedding vectors. All the values in a line are separated by a blank (' '). In practice, the embeddings can be learned by all deep neural language models.

> NOTE: Due to the large size of neural language models, we only upload the embedding vectors of words in the training and testing sets. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. trainPath: The path of the training set in the format of "word1 \t word2 \t label" triples. As for the label, 1 is for the hypernymy relation and 0 is for the non-hypernymy relation.

3. testPath: The path of the testing set. The format of the testing set is the same as that of the training set.

4. outputPath: The path of the output file, containing the model prediction scores of all the pairs in the testing set. The output of each pair is a real value in (-1,1). (Please refer to the paper for detailed explanation.)

5. dimension: The dimensionlaity of the embedding vectors.

> NOTE: The default values can be set are: "word_vectors.txt", "train.txt", "test.txt", "output.txt" and "50".

+ Eval: A simple evaluation script,  with three input argements required. It outputs Precision, Recall and F1-score  as the evaluation scores. 

1. truthPath: The path of the testing set, with human-labeled results.

2. predictPath: The path of the model output file,.

3. thres: A threshold in (-1,1) for the model to assign relation labels to Chinese word pairs. (Please refer to the parameter 'θ' in the paper.)

> NOTE: The default values can be set as: "test.txt", "output.txt" and "0.1".

**Dependencies**

1. This software is run in the JaveSE-1.8 environment. With a large probability, it runs properly in other versions of JaveSE as well. However, there is no guarantee.

2. It requires the FudanNLP toolkit for Chinese NLP analysis (https://github.com/FudanNLP/fnlp/), and the JAMA library for matrix computation (https://math.nist.gov/javanumerics/jama/). We use Jama-1.0.3.jar in this project.

**Citation**

If you find this software useful for your research, please cite the following paper.

> @inproceedings{acl2017,<br/>
&emsp;&emsp; author = {Chengyu Wang and Junchi Yan and Aoying Zhou and Xiaofeng He},<br/>
&emsp;&emsp; title = {Transductive Non-linear Learning for Chinese Hypernym Prediction},<br/>
&emsp;&emsp; booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},<br/>
&emsp;&emsp; pages = {1394–1404},<br/>
&emsp;&emsp; year = {2017}<br/>
}

More research works can be found here: https://chywang.github.io.



