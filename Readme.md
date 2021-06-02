<code>
  <h1 align="center">CANet: Cross-Disease Attention Network for
Joint Diabetic Retinopathy and Diabetic
Macular Edema Grading </h1>
</code>

<!-- For logo -->
<!-- <p align="center">
  <img src="https://github.com/shoheiyokoyama/Assets/blob/master/Gemini/logo.png" width="500">
</p> -->

# Overview

<!-- <img src="https://github.com/shoheiyokoyama/Assets/blob/master/Gemini/demo-circle-rotation.gif" align="left"> -->



Diabetic retinopathy (DR) and diabetic macular edema (DME) are the leading causes of permanent blindness in the working-age population.Automatic grading of DR and DME helps ophthalmologists design tailored treatments to patients, thus is of vital importance in the clinical practice. However, prior works either grade DR or DME, and ignore the correlation betweenDR and its complication,i.e ., DME.

# Features

![RepoSize](https://img.shields.io/github/repo-size/shaurya-src/Web-Automation?logo=GitHub&style=flat-square)
![License](https://img.shields.io/github/license/shaurya-src/Web-Automation?logo=GitLab&style=flat-square)
![LastCommit](https://img.shields.io/github/last-commit/shaurya-src/Web-Automation?logo=Git&style=flat-square)

<img src="https://media.giphy.com/media/xT0Gqn9yuw8hnPGn5K/giphy.gif" align="right" width="300" height="300">

# <a name="project-info"> Project Details

The Proposed Architecture In The Research Paper
  
First, we adopt a convolutional neural network, i.e., ResNet50 [42] to produce a set of feature maps with different resolutions. Then, we take the feature maps F ∈ RC×H×W with the smallest resolution and highly-semantic information (the deepest convolutional layer in ResNet50) as the inputs for the following two disease-specific attention modules, which learn the disease-specific features F∈ RC×H×W and Fj ∈ RC×H×W to understand each individual disease.
The disease-specific attention module to selectively learn useful features for individual diseases, and also design an effective disease-dependent attention module to capture the internal relationship between two diseases.
Afterward, we propose disease-dependent attention modules to explore the internal relationship between the two correlative diseases and produce the disease-dependent features for DR and DME, respectively. Finally, we predict the grading scores for DR and DME based on the learned disease-dependent features.
  

## <a name="requirements"> Requirements

- Tensorflow 2.1
- Keras 1.3
- Numpy
- Pandas
- Matplotlib
- Sklearn


## <a name="license"> License

*Project* is available under the MIT license. See the [LICENSE](https://github.com/shaurya-src/repo-template/blob/main/LICENSE) file for more info.

## <a name="author"> Author: Muskan Didwania
<!---
```python
# Muskan Didwania
```
-->

<p align="left">
  <code> Muskan Didwania </code>
</p>


<br>

