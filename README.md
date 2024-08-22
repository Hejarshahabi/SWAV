# Landslide Detection using SWAV and U-Net

This repository contains the implementation of a self-supervised learning model (SWAV) and a U-Net model for landslide detection using the Landslide4Sense competition dataset.


##                              SWAV Model Flowchart

![ SWAV Model Flowchart](https://github.com/Hejarshahabi/SWAV/blob/main/swav_model_flowchart1.jpg)
[<img src="image.png" width=2126/>](https://github.com/Hejarshahabi/SWAV/blob/main/swav_model_flowchart2.jpg)




## Project Structure

- **main_swav.py**: The main script for running the SWAV model.
- **UNet_train.py**: Script for training the U-Net model.
- **UNet_validation.py**: Script for validating the U-Net model.
- **src/**: Contains the following utility scripts:
  - **logger.py**: Logging utilities.
  - **multicropdataset.py**: Dataset handling for multi-crop augmentation.
  - **resnet.py**: ResNet model definition.
  - **utils.py**: General utilities used across the project.

## How to Run

### Prerequisites

- Python 3.x
- PyTorch
- 
### Training

To train the SWAV model, run:

```bash
python main_swav.py
```


