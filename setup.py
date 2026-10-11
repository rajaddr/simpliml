import setuptools, os, datetime, license

pckName = "simpliml"
version = "1.5.0"
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
<img src="https://i.ibb.co/jvM4fsVN/Simpli-ML.png"/>
</p><div align="center">

![Python](https://img.shields.io/pypi/pyversions/simpliml/{1}?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=python)
[![Version](https://img.shields.io/static/v1?label=Version&labelColor=007676&message={1}&color=01C0C0&style=flat)](https://pypi.org/project/simpliml/{1}/)
![PyPI - Version](https://img.shields.io/pypi/v/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
![Package Status](https://img.shields.io/static/v1?label=Status&labelColor=007676&message=Planning&color=01C0C0&style=flat)
[![License](https://img.shields.io/static/v1?label=License&labelColor=007676&message=MIT&color=01C0C0&style=flat)](https://github.com/rajaddr/simpliml/blob/master/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
[![Documentation Status](https://img.shields.io/readthedocs/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=readthedocs)](https://simpliml.readthedocs.io/en/latest/?badge=latest)
![Test Coverage](https://img.shields.io/static/v1?label=TestCoverage&labelColor=007676&message=100%&color=01C0C0&style=flat&logoColor=01C0C0&logo=pytest)

<hr>
</div>

**SimpliML** is a Python machine-learning library designed to simplify the
data lifecycle. The project brings data preparation, exploratory analysis,
forecasting, model evaluation, and result interpretation together in one
consistent workflow. It is intended for data scientists, analysts, students,
and developers who want practical forecasting tools without building the
entire modeling pipeline from scratch.

SimpliML currently focuses on time-series forecasting. It accepts historical
observations, prepares regular date-based datasets, evaluates multiple model
families, and returns comparable results for selecting a forecast. The package
is designed to work with familiar Python data tools, especially pandas
`DataFrame` objects.

## Key Features

- **Time-Series Data Preparation**  
  Convert date columns, generate regular date ranges, preserve historical
  observations, handle missing dates, add calendar features, and create future
  periods for prediction.

- **Data Analysis**  
  Explore trends, seasonality, distributions, autocorrelation, and partial
  autocorrelation with interactive and statistical visualizations.

- **Multiple Model Families**  
  Train statistical models through statsmodels, ensemble and regression models
  through scikit-learn, and neural-network forecasting models through PyTorch.

- **Flexible Execution**  
  Run model evaluation sequentially, with multiple threads, or with multiple
  processes depending on the workload and environment.

- **Model Comparison**  
  Compare successful models using MAE, MSE, RMSE, RMSLE, MAPE, R2, SMAPE,
  MFE, NMSE, and Theil's U statistic.

- **Forecast Result Review**  
  Select the best available model by MAPE or inspect a specific model, then
  combine historical, test, and forecast values for analysis and visualization.

## Forecasting Workflow

The package follows a simple end-to-end workflow:

1. **Prepare** historical date and target-value data.
2. **Generate** a regular time series and future dates.
3. **Analyze** trends, seasonality, and statistical behavior.
4. **Train** supported forecasting models with a selected seasonal period.
5. **Evaluate** model performance on held-out observations.
6. **Select** the best successful model and review its forecast.

The public time-series API includes `generateTSData`, `analysisData`,
`runModel`, `runThreadModel`, `runProcessModel`, `buildModel`, and
`modelResult`.

## Supported Data and Models

Input data is represented with pandas dataframes. The forecasting workflow uses
a date column and a numeric target column named `Data` after preparation.
Regular data frequencies such as daily, monthly, and quarterly observations
are supported.

The model pipeline can include:

- Statistical forecasting models from statsmodels, including ARIMA, AutoReg,
  exponential smoothing, Holt, SARIMA/SARIMAX, VAR, and VARMAX.
- Scikit-learn regression and ensemble estimators for feature-based
  forecasting.
- PyTorch LSTM-based forecasting models for neural-network workflows.

Model execution can be configured with `FAST` for a reduced search or `BEST`
for a broader search. The selected seasonal period should match the data
frequency, such as `7` for weekly seasonality in daily data or `12` for yearly
seasonality in monthly data.

## Intended Uses

SimpliML can support demand and sales forecasting, operational planning,
monthly or daily KPI prediction, time-series experimentation, classroom
projects, and rapid comparisons of statistical and machine-learning models.
It is a forecasting toolkit rather than a replacement for domain validation,
data quality checks, or production monitoring.

## Documentation

The official documentation is available on
[Read the Docs](https://simpliml.readthedocs.io/).

## Project Links

- [PyPI](https://pypi.org/project/simpliml/)
- [GitHub Repository](https://github.com/rajaddr/simpliml)
- [Issue Tracker](https://github.com/rajaddr/simpliml/issues)

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
    license="MIT",
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
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"

    ],
    #package_dir=package_dir,
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "*pycache*"]),
    #scripts=scripts,
    include_package_data=False,
    python_requires=python_requires,
    install_requires=install_requires,
)

# https://elmah.io/tools/base64-image-encoder/