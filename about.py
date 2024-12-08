import json

version_json = '''{
    "__date__": "2024-12-08 21:15:29.816579",
    "__author__": "Dharmaraj D",
    "__version__": "1.0.0",
    "__email__": "rajaddr@gmail.com",
    "__description__": "Machine Learning, Artificial Intelligence, Mathematics",
    "__keywords__": "Forecast Time Series Machine Learning Deep Learning Artificial Intelligence Mathematics",
    "__url__": "https://github.com/rajaddr/simpliml",
    "__license__": "MIT",
    "__status__": "Planning"
}'''

def get_about():
    return json.loads(version_json)
