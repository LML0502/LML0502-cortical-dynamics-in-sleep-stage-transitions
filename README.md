# Sample code for: Progressive cortical dynamics and underlying mechanisms in sleep stage transitions

### Introduction
This repository contains the code that accompanies 'Progressive cortical dynamics and underlying mechanisms in sleep stage transitions'. The purpose of the code in this repository is to provide examples of how to use the released data.

### Installation
```bash
git clone https://github.com/LML0502/LML0502-cortical-dynamics-in-sleep-stage-transitions.git
cd LML0502-cortical-dynamics-in-sleep-stage-transitions/paper_code

conda env create -n <your_env_name> -f environment.yml
conda activate <your_env_name>
pip install -r requirements.txt
```

### Embeddings Data
The data used in the article has been uploaded to the following websites and can be downloaded by yourself.
<a href="https://zenodo.org/uploads/15307159" target="data_link">data_link</a>

### Visualization and analysis

#### EEG data analysis:
```bash
python paper_main.py
```

#### The code of modeling:
Run the Jupyter notebook:
```bash
jupyter notebook sleep_dynamics_modeling.ipynb
```

### Project Structure
- `paper_main.py` - Main script for EEG data analysis and visualization
- `sleep_dynamics_modeling.ipynb` - Jupyter notebook for neural dynamics modeling
- `plot_figure.py` - Functions for generating figures
- `source_function.py` - Utility functions for data processing
- `so_spindle_counts.py` - Sleep spindle detection and analysis
- `network.py` - Network analysis functions
- `rate_model.py` - Rate-based neural modeling
- `test_statistics.py` - Statistical testing functions
- `requirements.txt` - Python package dependencies for pip installation
- `environment.yml` - Conda environment configuration

### Requirements
- Python 3.10 or higher
- See `requirements.txt` for detailed package dependencies
