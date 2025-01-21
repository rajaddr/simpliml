import json

version_json = '''{
    "__date__": "2025-01-21 11:50:22.085398",
    "__author__": "Dharmaraj D",
    "__version__": "1.0.32",
    "__email__": "rajaddr@gmail.com",
    "__description__": "Machine Learning, Artificial Intelligence, Mathematics",
    "__keywords__": "Supervised Learning, Neural Networks, Reinforcement Learning, Gradient Descent, TensorFlow, Probability Distributions, PCA, Convex Optimization, Transformers, Bayesian Inference, Markov Chains, GANs, SVD, Entropy, Attention Mechanism",
    "__url__": "https://github.com/rajaddr/simpliml",    
    "__license__": "MIT",
    "__status__": "Planning",
    "__doc__": "https://simpliml.readthedocs.io/"
}'''

def get_about():
    return json.loads(version_json)
