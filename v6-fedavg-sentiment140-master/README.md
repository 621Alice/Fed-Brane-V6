
<h3 align=center> A federated averaging learning algrithm for sentiment analysis</h3>

--------------------

<h4> v6-average-py </h4>

This algorithm is based on [documentation](https://docs.vantage6.ai/v/2.0.0/algorithm-development/create-new-algorithm). 
Vantage6 allowes to execute computations on federated datasets.
This algorithm uses Vantage6 in combination with PyTorch models to build a federated averaging learning pipeline for the binary sentiment classification of Sentiment140 dataset.


<h4> required package </h4>
Vantage6-client 2.1.0
<br>
torch 1.9.0
<br>
torchvision
<br>
pandas
<br>
numpy
<br>
scikit-learn 0.24.2
<br>
nltk
<br>
tqdm

<h4> how to run on mock client </h4>
pip install -e .
<br>
python example.py
<br>

<h4> how to run on real client </h4>
docker build -t sentiment .
<br>
python client.py

------------------------------------
> [vantage6](https://vantage6.ai)
