# 1. Introduction
Research Questions
Objectives
Contributions
Organization
# 2. Theoretical Background

## 2.1. SAR Imagery for Forest Parameter Estimation
## 2.2. Multitask Learning in Remote Sensing
## 2.3. Dataset Availibility in Forest Monitoring
# III Methodology

## A. Dataset

## B. Data Preprocessing

## C. Model Architecture

# IV Experiments and Results Analysis

## A. Experimental Configuration and Parameter Settings

1. set up in Colab, leveraging GPU A100, why (data size estimation)? what could be leverage in A100 (related works, reference)?

## B. Experimental Results

1. Baseline Models
    1. poor results in treesatai, better feature in UNet
    2. set benchmark for gen-chm regression

## C. Comparison Analysis

## D. Ablation Study

# V Conclusion


I could use another subsection here:

Yes, the whole thesis surrounds the topic and the title of "Multitask Learning for Extraction of Bio-/Geo Physical Parameters from SAR Imagery". First before the first subsection, write some background(quite short, since in the literature review parts, these would be addressed): trends in using deep learning in remote sensing modeling(compared to physical model and machine learning); SAR and optical imagery in forest monitoring for parameters like chm, biomass, tree height, soil moisture, dead/..trees.

1. motivations: This subsection should be combinition of background and motivations. You can form something from these perspectives(please adress my scope in remote sensing, deep learning, forest monitoring):

- SAR Imagery alone provides forest features that enough to predict/model tree parameters like canopy height(research for papers that holds), since the SAR features are ... (like not affect by weather, and so on, give some and related papers reference)
- existing research suggests the benefit of multi-task learning in remote sensing imagery(give several solid research proof); research gaps: not so much solely using feature extracted from SAR, but most of them leverage optical imagery.
- datasets: most dataset holds for single task in forest monitoring, not so much for multitask's perspectives. A motivation to research on dataset availability and quality, then generate a multitask dataset based on existing datasets.

1. research questions, please generate resonable contents and questions based on my opinion:

- Extending dataset...
- Can a single model simultaneously predict tree species, and biomass from Sentinel-1 SAR?
- Does multitask learning improve performance over single-task approaches?
- How do different fusion strategies affect the results as ablation/comparison studies?

1. I have no idea how to write objectives. If I already have motivation and research questions, is it necessary to write objectives?

2. My main task is tree species and canopy height, not biomass. Please change the relevant content.
3. I would like to address the dataset possibilities and assessments in the literature review section, so don't go too much here (like specific reference, don't do it), just general idea about the dataset extension
4. I said a master-student-like writing style and not AI-like, I find the paragraphs under introduction and 1.1.1, 1) address the issues too severe, 2) using the words that are not common and seemed to only used by AI, like alarmingly high, workhorses... 3) replace all the —, not just in those paragraphs but the whole draft.


 I said a master-student-like writing style and not AI-like, using common words, and replace all the —.