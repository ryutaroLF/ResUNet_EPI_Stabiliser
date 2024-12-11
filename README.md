# Misaligned Multi-View Correction Using ResUNet

This repository provides an implementation for correcting misaligned multi-view images using ResUNet, aligning them for further processing.

## Code Description

### Main Code
- **`main_v4_4_ResUNet_DWSC3.py`**: The main script for correcting misaligned multi-view images.
- **Execution Command**:
  ```bash
  python main_v4_4_ResUNet_DWSC3.py
  ```
- **Corresponding Job Script**: `job/job_v4_4.sh`

## Workflow

1. **Dataset**: Utilize the dataset from [Honauer2016](https://lightfield-analysis.uni-konstanz.de/).
2. **Generation of Misaligned Multi-View**:
   Artificially modify the arrangement of the dataset to generate misaligned multi-view images.
3. **Correction with ResUNet**:
   Use ResUNet to correct the misaligned images and align them.
4. **Integration with EPINET**:
   Feed the aligned multi-view images into a 1-stream EPINET for further processing.

## Acknowledgments

The evaluation code in this repository heavily utilizes parts of the implementation from:

**EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images**

Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon, and Seon Joo Kim  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2018*  
Repository: [https://github.com/chshin10/epinet](https://github.com/chshin10/epinet)
