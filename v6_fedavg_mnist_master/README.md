
<h3 align=center> A federated averaging learning algrithm classifying MINST dataset</h3>

--------------------

# v6-average-py

This algorithm is based on [documentation](https://docs.vantage6.ai/v/2.0.0/algorithm-development/create-new-algorithm). 
Vantage6 allowes to execute computations on federated datasets.
This algorithm uses Vantage6 in combination with PyTorch models to build a federated averaging learning pipeline for the classification of MNIST dataset.

# required package

Vantage6-client 2.1.0  
PyTorch 1.9.0  
Opacus 0.12.0  
scikit-learn 0.24.2  

# datasets
> [train_mnist.csv](https://www.python-course.eu/neural_network_mnist.php)

# how to test locally inside the directory

pip install -e .  
python example.py

# how to run with the mock client

if your are using the[train_mnist.csv](https://www.python-course.eu/neural_network_mnist.php), change the mock_client.py in line 17 to "pandas.read_csv(dataset,header=None)"

# how to run with the real clients

------------------------------------
> [vantage6](https://vantage6.ai)
