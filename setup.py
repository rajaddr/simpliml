import setuptools, os, datetime, license

pckName = "simpliml"
version = "1.4.0"
python_requires=">=3.9"
author = "Dharmaraj D"
status = "Development Status :: 1 - Planning"
author_email = "rajaddr@gmail.com"
licenseType="MIT"
description = "Machine Learning, Artificial Intelligence, Mathematics"
url = "https://github.com/rajaddr/simpliml"
docURL="https://simpliml.readthedocs.io/"
keywords='Supervised Learning, Neural Networks, Reinforcement Learning, Gradient Descent, TensorFlow, Probability Distributions, PCA, Convex Optimization, Transformers, Bayesian Inference, Markov Chains, GANs, SVD, Entropy, Attention Mechanism'

with open('LICENSE', 'w') as f:
    f.write(license.find(licenseType).render(name=author, email=author_email))

with open('simpliml/about.py', 'w') as f:
    f.write("""import json

version_json = '''{
    "__date__": "%s",
    "__author__": "%s",
    "__version__": "%s",
    "__email__": "%s",
    "__description__": "%s",
    "__keywords__": "%s",
    "__url__": "%s",    
    "__license__": "%s",
    "__status__": "%s",
    "__doc__": "%s"
}'''

def get_about():
    return json.loads(version_json)
"""%(datetime.datetime.now(),author, version, author_email, description, keywords, url, licenseType, status.split()[-1], docURL))

with open('README.md', 'w') as f:
    f.write("""<p align="center">
<img src="https://i.ibb.co/cKFYB1xq/Simpli-ML.png"/>
</p><div align="center">

![Python](https://img.shields.io/pypi/pyversions/simpliml/{1}?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=python)
[![Version](https://img.shields.io/static/v1?label=Version&labelColor=007676&message={1}&color=01C0C0&style=flat)](https://pypi.org/project/simpliml/{1}/)
![PyPI - Version](https://img.shields.io/pypi/v/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
![Package Status](https://img.shields.io/static/v1?label=Status&labelColor=007676&message=Planning&color=01C0C0&style=flat)
[![License](https://img.shields.io/static/v1?label=License&labelColor=007676&message=MIT&color=01C0C0&style=flat)](https://github.com/rajaddr/simpliml/blob/master/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
[![Documentation Status](https://img.shields.io/readthedocs/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=readthedocs)](https://simpliml.readthedocs.io/en/latest/?badge=latest)
![Test Coverage](https://img.shields.io/static/v1?label=TestCoverage&labelColor=007676&message=89%&color=01C0C0&style=flat&logoColor=01C0C0&logo=pytest)

<hr>
</div>

**SimpliML** is a versatile machine learning library designed to be a one-stop solution for the entire data lifecycle. Whether you're preparing raw data or deploying advanced predictive models, *SimpliML* simplifies every step of the machine learning process.  

## Key Features  

- **Data Cleansing and Cleaning**  
  Simplify the preprocessing of raw data to ensure accurate and reliable model performance.  

- **Data Analysis**  
  Explore and analyze data with powerful, easy-to-use tools to uncover actionable insights.  

- **Model Execution and Prediction**  
  Train, validate, and deploy machine learning models seamlessly for accurate and efficient predictions.  

- **Forecasting and Optimization**  
  Perform precise forecasting and optimize your decision-making processes with ease.  

## Why Choose SimpliML?  

***SimpliML*** is designed for data scientists, ML engineers, and enthusiasts who need a reliable, efficient, and easy-to-use toolkit for managing the entire machine learning workflow.  

Get started today and unlock the full potential of your data with **SimpliML**!

- **Time Series**
- **Many more comming soon**

## Documentation
The official documentation is hosted on [Click Here](https://simpliml.readthedocs.io/).

""".format(python_requires, version, status.split()[-1], licenseType, url))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = list(filter(None, [i.strip() for i in open("C:\\Users\\raja_\\OneDrive\\Projects\\pipTSF\\simplimlPkg\\requirements.txt", "r", encoding="utf-8").readlines()]))

package_dir = {pckName: pckName}
packages = []
scripts = []


for dirname, dirnames, filenames in os.walk(os.getcwd()):
    if '__init__.py' in filenames:
        packages.append(  dirname.replace(os.getcwd(), '').replace('/', '.').replace('\\', '.'))

for dirname, dirnames, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        if filename.endswith('.py'):
            scripts.append(os.path.join(dirname, filename))

setuptools.setup(
    name=pckName,
    version=version,
    author=author,
    author_email=author_email,
    setup_requires=["datetime", "license", "setuptools"],
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    keywords = keywords,
    project_urls={
        "Issue Tracker": url + "/issues",
        "Changelog": url + "/blob/master/CHANGELOG.md",
        "Documentation": docURL,
        "Repository": url
    },
    classifiers=[
        status,
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        license.find(licenseType).python,
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"

    ],
    #package_dir=package_dir,
    packages=setuptools.find_packages(exclude=["*pycache*"]),
    #scripts=scripts,
    include_package_data=False,
    python_requires=python_requires,
    install_requires=install_requires,
)

# https://elmah.io/tools/base64-image-encoder/