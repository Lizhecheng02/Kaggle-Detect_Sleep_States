## This Repo is For [Kaggle Child Mind Institute - Detect Sleep States Competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)

## Detail Discussion: [Here](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459616)

## PrecTime Model
### Reproduce According to [PrecTime: A Deep Learning Architecture for Precise Time Series Segmentation in Industrial Manufacturing Operations](https://arxiv.org/abs/2302.10182)

- PrecTime.py

  **Base model with constant parameters**

- PrecTime - LSTM

  **Use LSTM models for Inter-Window Context Detection**

- PrecTime - GRU

  **Use GRU models for Inter-Window Context Detection**

- PrecTime - Transformer

  **Replace LSTM by Transformer for Inter-Window Context Detection**

- PrecTime - Final

  **Combine LSTM and Transformer + Adjustable parameters + Automatic parameter calculation**
