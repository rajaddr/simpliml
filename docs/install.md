# Installation

SimpliML is distributed on [PyPI](https://pypi.org/project/simpliml/) and
supports Python 3.9 and newer. A virtual environment is recommended so that
SimpliML's dependencies do not affect other Python projects.

## Install from PyPI

Create and activate a virtual environment, then install the latest release:

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install simpliml
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install simpliml
```

If PowerShell does not allow activation scripts, enable them for the current
user or activate the environment from Command Prompt instead:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```bat
.venv\Scripts\activate.bat
```

## Verify the installation

Run this command while the virtual environment is active:

```bash
python -c "import simpliml; print(simpliml.__version__)"
```

The command should print the installed package version. To verify the
time-series forecasting API as well:

```bash
python -c "import simpliml.timeseriesforecast as tsf; print(tsf.generateTSData)"
```

## Install a specific version

To install the documented 1.5.0 release explicitly:

```bash
python -m pip install "simpliml==1.5.0"
```

Upgrade an existing installation with:

```bash
python -m pip install --upgrade simpliml
```

## Dependencies

Installing SimpliML from PyPI installs its runtime dependencies:

| Package | Purpose |
| --- | --- |
| pandas | Dataframe and time-series operations |
| SciPy | Scientific computing |
| scikit-learn | Machine-learning models and metrics |
| statsmodels | Statistical forecasting models |
| Altair | Interactive data visualization |
| Matplotlib | Statistical and model plots |
| Seaborn | Statistical visualization |
| PyTorch | Neural-network forecasting models |

PyTorch wheels vary by operating system and hardware. If you need a
CUDA-specific build, follow the installation command provided by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing or reinstalling SimpliML.

## Install from a source checkout

Clone the repository and install it in editable mode when developing:

```bash
git clone https://github.com/rajaddr/simpliml.git
cd simpliml
python -m venv .venv
```

Activate `.venv` using the instructions above, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

After installation, the package can be imported normally:

```python
import simpliml
import simpliml.timeseriesforecast as tsf
```

## Troubleshooting

- Check that the active interpreter is the one in `.venv`:
  `python -c "import sys; print(sys.executable)"`.
- Use `python -m pip` rather than a standalone `pip` command to ensure that
  packages are installed into the active environment.
- If dependency resolution fails, upgrade pip and retry:
  `python -m pip install --upgrade pip`.
- If a PyTorch installation is incompatible with your hardware, install the
  appropriate wheel from the official PyTorch instructions and retry the
  SimpliML installation.

For usage examples, continue to the
[Time Series guide](tsf.md).
