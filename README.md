# Awesome-Multimodal-Adaptation

## Table of Contents
- [Awesome-Multimodal-Adaptation](#awesome-multimodal-adaptation)
  - [Table of Contents](#table-of-contents)
  - [Multimodal Domain Adaptation](#multimodal-domain-adaptation)
    - [Action Recognition](#action-recognition)
    - [3D Semantic Segmentation](#3d-semantic-segmentation)
    - [Other Tasks](#other-tasks)
  - [Multimodal Test-time Adaptation](#multimodal-test-time-adaptation)
    - [Action Recognition](#action-recognition-1)
    - [3D Semantic Segmentation](#3d-semantic-segmentation-1)
    - [Other Tasks](#other-tasks-1)
  - [Multimodal Domain Generalization](#multimodal-domain-generalization)
    - [Action Recognition](#action-recognition-2)
    - [3D Semantic Segmentation](#3d-semantic-segmentation-2)
  - [Domain Adaptation and Generalization with the Help of Multimodal Foundation Models](#domain-adaptation-and-generalization-with-the-help-of-multimodal-foundation-models)
  - [Adaptation of Foundation Models](#adaptation-of-foundation-models)




## Multimodal Domain Adaptation

### Action Recognition

(ACM MM 2018) [A Unified Framework for Multimodal Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3240508.3240633) by Qi et al.


(CVPR 2020) [Multi-Modal Domain Adaptation for Fine-Grained Action Recognition](https://arxiv.org/abs/2001.09691) [[Code]](https://github.com/jonmun/MM-SADA-code) by Munro et al.

(CVPR 2021) [Spatio-temporal Contrastive Domain Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_Spatio-temporal_Contrastive_Domain_Adaptation_for_Action_Recognition_CVPR_2021_paper.pdf) by Song et al.

(ICCV 2021) [Learning Cross-modal Contrastive Features for Video Domain Adaptation](https://arxiv.org/abs/2108.11974) by Kim et al.

(TIP 2021) [Progressive Modality Cooperation for Multi-Modality Domain Adaptation](https://ieeexplore.ieee.org/document/9334409) by Zhang et al.


(ACM MM 2021) [Differentiated Learning for Multi-Modal Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475660) by Lv et al.



(CVPR 2022) [Audio-Adaptive Activity Recognition Across Video Domains](https://arxiv.org/abs/2203.14240) [[Code]](https://github.com/xiaobai1217/DomainAdaptation) by Zhang et al.

(CVPR 2022) [Interact before Align: Leveraging Cross-Modal Knowledge for Domain Adaptive Action Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Interact_Before_Align_Leveraging_Cross-Modal_Knowledge_for_Domain_Adaptive_Action_CVPR_2022_paper.pdf) by Yang et al.

(ACM MM 2022) [Relative Alignment Network for Source-Free Multimodal Video Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3503161.3548009) by Huang et al.

(ACM MM 2022) [Mix-DANN and Dynamic-Modal-Distillation for Video Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3503161.3548313) by Yin et al.

(ECCV 2024) [Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision](https://arxiv.org/abs/2407.01518) [[Code]](https://github.com/donghao51/MOOSA) by Dong et al.



### 3D Semantic Segmentation
(CVPR 2020) [xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/1911.12676) [[Code]](https://github.com/valeoai/xmuda) by Jaritz et al.

(ICCV 2021) [Sparse-to-dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/2107.14724) [[Code]](https://github.com/leolyj/DsCML) by Peng et al.

(ISPRS 2021) [Adversarial unsupervised domain adaptation for 3D semantic segmentation with multi-modal learning](https://www.sciencedirect.com/science/article/pii/S0924271621001131) by Liu et al.

(ECCV 2022) [Drive&Segment: Unsupervised Semantic Segmentation of Urban Scenes via Cross-modal Distillation](https://arxiv.org/abs/2203.11160) [[Code]](https://github.com/vobecant/DriveAndSegment) by Vobecky et al.

(TPAMI 2022) [Cross-modal Learning for Domain Adaptation
in 3D Semantic Segmentation](https://arxiv.org/abs/2101.07253) [[Code]](https://github.com/valeoai/xmuda_journal) by Jaritz et al.

(ACM MM 2022) [Cross-Domain and Cross-Modal Knowledge Distillation in Domain Adaptation for 3D Semantic Segmentation](https://dl.acm.org/doi/pdf/10.1145/3503161.3547990) by Li et al.

(ACM MM 2022) [Self-supervised Exclusive Learning for 3D Segmentation with Cross-modal Unsupervised Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3503161.3547987) by Zhang et al.

(ICCV 2023) [CrossMatch: Source-Free Domain Adaptive Semantic Segmentation via Cross-Modal Consistency Training](https://openaccess.thecvf.com/content/ICCV2023/papers/Yin_CrossMatch_Source-Free_Domain_Adaptive_Semantic_Segmentation_via_Cross-Modal_Consistency_Training_ICCV_2023_paper.pdf) by Yin et al.

(ICCV 2023) [SUMMIT: Source-Free Adaptation of Uni-Modal Models to Multi-Modal Targets](https://openaccess.thecvf.com/content/ICCV2023/papers/Simons_SUMMIT_Source-Free_Adaptation_of_Uni-Modal_Models_to_Multi-Modal_Targets_ICCV_2023_paper.pdf) [[Code]](https://github.com/csimo005/SUMMIT) by Simons et al.


(AAAI 2023) [Cross-Modal Contrastive Learning for Domain Adaptation in 3D Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/download/25400/25172) by Xing et al.

(RAS 2023) [Real-time multi-modal semantic fusion on unmanned aerial vehicles with label propagation for cross-domain adaptation](https://www.sciencedirect.com/science/article/pii/S0921889022001750) by Bultmann et al.

(CVPRW 2023) [Exploiting the Complementarity of 2D and 3D Networks to Address Domain-Shift in 3D Semantic Segmentation](https://arxiv.org/abs/2304.02991) by Cardace et al.


(IROS 2023) [DualCross: Cross-Modality Cross-Domain Adaptation for Monocular BEV Perception](https://arxiv.org/abs/2305.03724) [[Code]](https://github.com/YunzeMan/DualCross) by Man et al.

(ACM MM 2023) [Cross-modal & Cross-domain Learning for Unsupervised LiDAR Semantic Segmentation](https://dl.acm.org/doi/pdf/10.1145/3581783.3612320) by Chen et al.

(ACM MM 2023) [Cross-modal Unsupervised Domain Adaptation for 3D Semantic Segmentation via Bidirectional Fusion-then-Distillation](https://dl.acm.org/doi/pdf/10.1145/3581783.3612013) by Wu et al.

(AAAI 2023) [Mx2M: Masked Cross-Modality Modeling in Domain Adaptation for 3D Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/25448) by Zhang et al.

(ICRA 2024) [MoPA: Multi-Modal Prior Aided Domain Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/2309.11839) [[Code]](https://github.com/AronCao49/MoPA) by Cao et al.




### Other Tasks

(TMM 2019) [Deep Multi-Modality Adversarial Networks for Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/document/8656504) by Ma et al.

(CVPR 2023) [OSAN: A One-Stage Alignment Network to Unify Multimodal Alignment and Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_OSAN_A_One-Stage_Alignment_Network_To_Unify_Multimodal_Alignment_and_CVPR_2023_paper.pdf) by Liu et al.

(ACL 2024) [Amanda: Adaptively Modality-Balanced Domain Adaptation for Multimodal Emotion Recognition](https://aclanthology.org/2024.findings-acl.859/) by Zhang et al.

(WACVW 2024) [Source-Free Domain Adaptation for RGB-D Semantic Segmentation with Vision Transformers](https://arxiv.org/abs/2305.14269) by Rizzoli et al.




## Multimodal Test-time Adaptation
### Action Recognition

(ICLR 2024) [Test-time Adaptation against Multi-modal Reliability Bias](https://openreview.net/forum?id=TPZRq4FALB) [[Code]](https://github.com/XLearning-SCU/2024-ICLR-READ) by Yang et al.

(CVPR 2024) [Modality-Collaborative Test-Time Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.pdf) by Xiong et al.

(ICMLW 2024) [Two-Level Test-Time Adaptation in Multimodal Learning](https://openreview.net/forum?id=n0lDbIKVAT) by Lei et al.



### 3D Semantic Segmentation

(CVPR 2022) [MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/2204.12667) by Shin et al.

(CVPR 2023) [Multi-Modal Continual Test-Time Adaptation for 3D Semantic Segmentation](https://arxiv.org/abs/2303.10457) by Cao et al.

(ECCV 2024) [Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation](https://arxiv.org/abs/2403.06461) [[Code]](https://github.com/AronCao49/Latte/) by Cao et al.


### Other Tasks
(AAAI 2024) [Heterogeneous Test-Time Training for Multi-Modal Person Re-identification](https://ojs.aaai.org/index.php/AAAI/article/view/28398) by Wang et al.

(CVPR 2024) [Test-Time Adaptation for Depth Completion](https://arxiv.org/abs/2402.03312) [[Code]](https://github.com/seobbro/TTA-depth-completion) by Park et al.


## Multimodal Domain Generalization
### Action Recognition
(WACV 2022) [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf) by Planamente et al.

(NeurIPS 2023) [SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization](https://arxiv.org/abs/2310.19795) [[Code]](https://github.com/donghao51/SimMMDG) by Dong et al.


(ECCV 2024) [Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision](https://arxiv.org/abs/2407.01518) [[Code]](https://github.com/donghao51/MOOSA) by Dong et al.

(IJCV 2024) [Relative Norm Alignment for Tackling Domain Shift in Deep Multi-modal Classification](https://link.springer.com/article/10.1007/s11263-024-01998-9) by Planamente et al.


### 3D Semantic Segmentation
(ICCV 2023) [BEV-DG: Cross-Modal Learning under Bird’s-Eye View for Domain Generalization of 3D Semantic Segmentation](https://arxiv.org/abs/2308.06530) by Li et al.



## Domain Adaptation and Generalization with the Help of Multimodal Foundation Models

(ICCV 2023) [The Unreasonable Effectiveness of Large Language-Vision Models for Source-free Video Domain Adaptation](https://arxiv.org/abs/2308.09139) [[Code]](https://github.com/giaczara/dallv) by Zara et al.

(ICCV 2023) [A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_A_Sentence_Speaks_a_Thousand_Images_Domain_Generalization_through_Distilling_ICCV_2023_paper.pdf) by Huang et al.

(ICCV 2023) [PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization](https://openaccess.thecvf.com/content/ICCV2023/papers/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.pdf) by Cho et al.



(CVPR 2024) [PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_PracticalDG_Perturbation_Distillation_on_Vision-Language_Models_for_Hybrid_Domain_Generalization_CVPR_2024_paper.pdf) by Chen et al.


(CVPR 2024) [Collaborating Foundation Models for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2312.09788) [[Code]](https://github.com/yasserben/CLOUDS) by Benigmim et al.

(CVPR 2024) [Disentangled Prompt Representation for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_Disentangled_Prompt_Representation_for_Domain_Generalization_CVPR_2024_paper.pdf) by Cheng et al.


(CVPR 2024) [Source-Free Domain Adaptation with Frozen Multimodal Foundation Model](https://arxiv.org/abs/2311.16510) [[Code]](https://github.com/tntek/source-free-domain-adaptation) by Tang et al.


(CVPR 2024) [Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation](https://arxiv.org/abs/2312.04265) [[Code]](https://github.com/w1oves/Rein) by Wei et al

(CVPR 2024) [Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation](https://arxiv.org/abs/2312.03502) by Zhang et al

(CVPR 2024) [Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification](https://openaccess.thecvf.com/content/CVPR2024/papers/Addepalli_Leveraging_Vision-Language_Models_for_Improving_Domain_Generalization_in_Image_Classification_CVPR_2024_paper.pdf) by Addepalli et al



(CVPRW 2024) [Exploring the Benefits of Vision Foundation Models for Unsupervised Domain Adaptation](https://arxiv.org/abs/2406.09896) [[Code]](https://github.com/tue-mps/vfm-uda) by Englert et al



(ECCV 2024) [DGInStyle: Domain-Generalizable Semantic Segmentation with Image Diffusion Models and Stylized Semantic Control](https://arxiv.org/abs/2312.03048) [[Code]](https://github.com/prs-eth/DGInStyle) by Jia et al.


(ECCV 2024) [Learning to Adapt SAM for Segmenting Cross-domain Point Clouds](https://arxiv.org/abs/2310.08820) by Peng et al.

(ECCV 2024) [Learning Representations from Foundation Models for Domain Generalized Stereo Matching](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05841.pdf) by Zhang et al.

(ECCV 2024) [Cross-Domain Semantic Segmentation on Inconsistent Taxonomy using VLMs](https://www.arxiv.org/abs/2408.02261) by Lim et al.



(ICMLW 2024) [Leveraging Generative Foundation Models for Domain Generalization](https://openreview.net/pdf?id=AycNSoRjbi) by Hemati et al.

(WACV 2024) [Empowering Unsupervised Domain Adaptation with Large-scale Pre-trained Vision-Language Models](https://openaccess.thecvf.com/content/WACV2024/papers/Lai_Empowering_Unsupervised_Domain_Adaptation_With_Large-Scale_Pre-Trained_Vision-Language_Models_WACV_2024_paper.pdf) by Lai et al.

(WACV 2024) [ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation](https://openaccess.thecvf.com/content/WACV2024/papers/Hu_ReCLIP_Refine_Contrastive_Language_Image_Pre-Training_With_Source_Free_Domain_WACV_2024_paper.pdf) by Hu et al.



(IJCV 2024) [Source-Free Domain Adaptation Guided by Vision and Vision-Language Pre-Training](https://arxiv.org/abs/2405.02954) [[Code]](https://github.com/zwenyu/colearn-plus) by Zhang et al.


(arXiv 2022) [Prompt Vision Transformer for Domain Generalization](https://arxiv.org/abs/2208.08914) [[Code]](https://github.com/zhengzangw/DoPrompt) by Zheng et al.

(arXiv 2024) [Visual Foundation Models Boost Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation](http://arxiv.org/abs/2403.10001) [[Code]](https://github.com/EtronTech/VFMSeg) by Xu et al.

(arXiv 2024) [Open-Set Domain Adaptation with Visual-Language Foundation Models](https://arxiv.org/abs/2307.16204) by Yu et al.

(arXiv 2024) [CLIP the Divergence: Language-guided Unsupervised Domain Adaptation](https://arxiv.org/abs/2407.01842) by Zhu et al.

(arXiv 2024) [Transitive Vision-Language Prompt Learning for Domain Generalization](https://arxiv.org/abs/2404.18758) by Chen et al.


## Adaptation of Foundation Models 

(ICCV 2023) [Black Box Few-Shot Adaptation for Vision-Language models](https://openaccess.thecvf.com/content/ICCV2023/papers/Ouali_Black_Box_Few-Shot_Adaptation_for_Vision-Language_Models_ICCV_2023_paper.pdf) by Ouali et al

(ICCV 2023) [Distilling Large Vision-Language Model with Out-of-Distribution Generalizability](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.pdf) [[Code]](https://github.com/xuanlinli17/large_vlm_distillation_ood) by Li et al.

(NeurIPS 2023) [Benchmarking robustness of adaptation methods on pre-trained vision-language models](https://proceedings.neurips.cc/paper_files/paper/2023/file/a2a544e43acb8b954dc5846ff0d77ad5-Paper-Datasets_and_Benchmarks.pdf) [[Code]](https://github.com/adarobustness/adaptation_robustness) by Chen et al


(CVPR 2024) [Efficient Test-Time Adaptation of Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.pdf) by Karmanov et al

(TIP 2024) [Adapting Vision-Language Models via Learning to Inject Knowledge](https://ieeexplore.ieee.org/abstract/document/10704586) by Xuan et al

