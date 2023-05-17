[TOC]

## 项目落地
| 年份 | 来源 | 名称 | 源码 | 数据集 | 描述 |
| :-----: | :-----:| :-----:| :-----: | :-----: | ------- |
| 2021 | FGCS |[Event prediction based on evolutionary event ontology knowledge](https://www.sciencedirect.com/science/article/abs/pii/S0167739X20311778)|[Code](https://github.com/hummingg/KGEvetPred)|[Data](https://github.com/hummingg/EEOK)|一个从事件**提取**到事件**预测**的**流水线**过程框架。考虑到不同的事件域，我们提供了一种领域感知的事件预测方法。|
| 2022 | NAACL |[RESIN-11: Schema-guided Event Prediction for 11 Newsworthy Scenarios](https://paperswithcode.com/paper/resin-11-schema-guided-event-prediction-for)|[Code](https://github.com/RESIN-KAIROS/RESIN-11)|[LDC](https://catalog.ldc.upenn.edu/)|schema-guided event extraction&prediction framework.(1) an open-domain end-to-end multimedia multilingual information extraction system with weak-supervision and zero-shot learningbased techniques. (2) schema matching and schema-guided event prediction based on our curated schema library.**Dockerized**|

# 事件抽取与事件关系抽取论文整理
## ChatGPT与研究方向
| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2023 |  |[Zero-Shot Information Extraction via Chatting with ChatGPT](https://readpaper.com/pdf-annotate/note?pdfId=4725954968591269889)|||

## Prompt学习

| 年份 | 来源 |                             名称                             |                     源码                      | 数据集 |
| :--: | :--: | :----------------------------------------------------------: | :-------------------------------------------: | :----: |
| 2021 | ACM  | [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://readpaper.com/paper/3185341429) |                                               |        |
| 2021 |      | [GPT Understands, Too](https://readpaper.com/paper/3139080614) |   [Code](https://github.com/THUDM/P-tuning)   |        |
|      |      | [PromptBERT: Improving BERT Sentence Embeddings with Prompts](https://readpaper.com/paper/625656177011560448) | [Code](https://github.com/kongds/Prompt-BERT) |        |
| 2022 |      | [Generating Disentangled Arguments with Prompts: A Simple Event Extraction Framework That Works](https://readpaper.com/paper/3205954317) |  [Code](https://github.com/RingBDStack/GDAP)  |        |
|      |      | [Prompt for Extraction? PAIE: Prompting Argument Interaction for Event Argument Extraction](https://readpaper.com/paper/4595124901795340289) |  [Code](https://github.com/mayubo2333/PAIE)   |        |
|      |      | [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://readpaper.com/paper/4551503280291717121) | [Code](https://github.com/THUDM/P-tuning-v2)  |        |
|      |      | [五万字综述！Prompt Tuning：深度解读一种新的微调范式](https://redian.news/wxnews/341131) |                                               |        |

## 事件关系抽取

### 事件因果关系抽取
| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2020 | COLING |[KnowDis: Knowledge Enhanced Data Augmentation for Event Causality Detection via Distant Supervision](https://www.aclweb.org/anthology/2020.coling-main.135)||[COPA](https://people.ict.usc.edu/~gordon/copa.html)|
| 2020 | IJCAI | [Knowledge Enhanced Event Causality Identification with Mention Masking Generalizations](https://www.ijcai.org/proceedings/2020/499) | [Code](https://github.com/jianliu-ml/EventCausalityIdentification) ||
| 2022 | SIGIR | [Towards Event-level Causal Relation Identification](https://doi.org/10.1145/3477495.3531758) | [Code](https://github.com/HITSZ-HLT/Graph4ECI) | [EventStoryLine](https://github.com/tommasoc80/EventStoryLine) |
### 事件共指关系抽取
| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2021 | AAAI | [Span-Based Event Coreference Resolution](https://www.aaai.org/AAAI21Papers/AAAI-9086.LJ.pdf) |||
| 2021 | EACL | [Automatic Data Acquisition for Event Coreference Resolution](https://aclanthology.org/2021.eacl-main.101/) | [Code](https://github.com/prafulla77/Event-Coref-EACL-2021) | [Data](https://drive.google.com/drive/folders/1NNBKiO4eYkGBjkdXGUieKg2fCWfbUBuf) |
| 2021 | NAACL | [A Context-Dependent Gated Module for Incorporating Symbolic Semantics into Event Coreference Resolution](http://arxiv.org/abs/2104.01697) | [Code](https://github.com/laituan245/eventcoref) | ACE 2005&KBP 2016 |
### 事件时序关系抽取
| 年份 | 来源 | 名称 | 源码 | 数据集 | 描述 |
| :-----: | :-----:| :-----:| :-----: | :-----: | :-----: |
| 2022 | COLING | [RSGT: Relational Structure Guided Temporal Relation Extraction](https://aclanthology.org/2022.coling-1.174) ||TBD & MATRES|使用的是GGNN(Gated Graph Neural Network)|
| 2022 | COLING | [DCT-Centered Temporal Relation Extraction](https://aclanthology.org/2022.coling-1.182) ||TBD & TDD-Man & TDD-Auto|使用的是GCN|
| 2022 | Computational Intelligence and Neuroscience | [Temporal Relation Extraction with Joint Semantic and Syntactic Attention](https://www.hindawi.com/journals/cin/2022/5680971/) | | TimeBank-Dense & MATRES | |
| 2021 | IJCAI | [Discourse-Level Event Temporal Ordering with Uncertainty-Guided Graph Completion](https://www.ijcai.org/proceedings/2021/533) | [Code](https://github.com/jianliu-ml/EventTemp) | TBD & TDD-Man & TDD-Auto | 第一个将图表示学习应用到篇章级事件时序关系抽取任务当中来的论文 |
| 2021 | IJCNLP | [TIMERS: Document-level Temporal Relation Extraction](https://aclanthology.org/2021.acl-short.67.pdf) |  |  |
| 2021 | [计算机研究与发展](https://crad.ict.ac.cn/) | [融合上下文信息的篇章级事件时序关系抽取方法](https://crad.ict.ac.cn/CN/abstract/abstract4529.shtml) |  |TimeBank-Dense & MATRES  |
| 2021 | NAACL |[EventPlus: A Temporal Event Understanding Pipeline](https://arxiv.org/abs/2101.04922) | [Code](https://github.com/PlusLabNLP/EventPlus) | TimeBank-Dense, MATRES & ACE2005 | 事件抽取与事件时序关系抽取的流水线模型 |
| 2019 | ACL | [Fine-Grained Temporal Relation Extraction](https://aclanthology.org/P19-1280.pdf) | [decomp](http://decomp.io/projects/time/) | [decomp](http://decomp.io/projects/time/) |
| 2019 | CCF | [Event Temporal Relation Classification Based on Graph Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-030-32236-6_35) |  | TimeBank-Dense ||
| 2019 | CONLL | [Deep Structured Neural Network for Event Temporal Relation Extraction](https://aclanthology.org/K19-1062/) | [Code](https://github.com/PlusLabNLP/Deep-Structured-EveEveTemp) | TCR, TimeBank-Dense & MATRES ||
| 2019 | EMNLP-IJCNLP | [Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction](https://aclanthology.org/D19-1041/) | [Code](https://github.com/PlusLabNLP/JointEventTempRel) | TB-Dense & MATRES ||
| 2018 |  | [TEMPROB: Improving Temporal Relation Extraction with a Globally Acquired Statistical Resource-ReadPaper](https://readpaper.com/paper/2797731290) |  |  |  provides prior knowledge of the temporal order that some events usually follow.|

### 事件父子关系抽取
| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2022 | TACL | [Decomposing and Recomposing Event Structure](https://aclanthology.org/2022.tacl-1.2.pdf) | [decomp](http://decomp.io/projects/event-structure/) | [decomp](http://decomp.io/projects/event-structure/) |
| 2021 | EMNLP | [Learning Constraints and Descriptive Segmentation for Subevent Detection](https://cogcomp.seas.upenn.edu/page/publication_view/950) | [CogComp/Subevent_EventSeg](https://github.com/CogComp/Subevent_EventSeg) | [HiEve](https://github.com/CogComp/Subevent_EventSeg/tree/main/hievents_v2) and [IC](https://github.com/CogComp/Subevent_EventSeg/tree/main/IC) |
| 2021 | EMNLP | [Weakly Supervised Subevent Knowledge Acquisition](https://aclanthology.org/2020.emnlp-main.430.pdf) | [SubeventAcquisition](https://github.com/wenlinyao/EMNLP20-SubeventAcquisition) | [RED, ESC, HiEve, Timebank](https://github.com/wenlinyao/EMNLP20-SubeventAcquisition/tree/master/datasets) and [RED](https://catalog.ldc.upenn.edu/LDC2016T23) |
| 2020 | EMNLP | [Joint Constrained Learning for Event-Event Relation Extraction](https://cogcomp.seas.upenn.edu/page/publication_view/914) | [CogComp/JointConstrainedLearning](https://github.com/CogComp/JointConstrainedLearning) | [MATRES](https://github.com/why2011btv/JointConstrainedLearning/tree/master/MATRES) and [HiEve](https://github.com/why2011btv/JointConstrainedLearning/tree/master/hievents_v2) |
## 事件抽取

### 中文（句子）事件抽取

| 年份 | 来源 |                             名称                             |                            源码                             |                            数据集                            |
| :--: | :--: | :----------------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------------: |
| 2020 | CCL  | [A Novel Joint Framework for Multiple Chinese Events Extraction](https://aclanthology.org/2020.ccl-1.88/) | [Code](https://github.com/prafulla77/Event-Coref-EACL-2021) | [Data](https://drive.google.com/drive/folders/1NNBKiO4eYkGBjkdXGUieKg2fCWfbUBuf) |
| 2021 | ACL  | [CasEE: A Joint Learning Framework with Cascade Decoding for Overlapping Event Extraction](https://aclanthology.org/2021.findings-acl.14/) |        [Code](https://github.com/JiaweiSheng/CasEE)         | [FewFC (中国金融事件提取数据集)](https://github.com/TimeBurningFish/FewFC) |
| 2022 | CAEE | [Chinese Event Extraction via Graph Attention Network \| ACM Transactions on Asian and Low-Resource Language Information Processing](https://dl.acm.org/doi/10.1145/3494533) |                                                             |                          自制数据集                          |

### 篇章级事件抽取

| 年份 |     来源      |                             名称                             |                             源码                             |                            数据集                            |
| :--: | :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2018 |      ACL      | [DCFEE: A Document-level Chinese Financial Event Extraction System based on Automatically Labeled Training Data](https://aclanthology.org/P18-4009/) |         [Code](https://github.com/yanghang111/DCFEE)         |         [Data](https://github.com/yanghang111/DCFEE)         |
| 2019 |     EMNLP     | [Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction](https://aclanthology.org/D19-1032/) |        [Code](https://github.com/dolphin-zs/Doc2EDAG)        | [ChFinAnn(2008年至2018年中国上市公司的财务公告)](https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip) |
| 2021 |      ACL      | （GIT）[Document-level Event Extraction via Heterogeneous Graph-based Interaction Model with a Tracker](https://aclanthology.org/2021.acl-long.274/) |           [Code](https://github.com/RunxinXu/GIT)            | [ChFinAnn](https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip) |
| 2021 |      ACL      | (DE-PPN)[Document-level Event Extraction via Parallel Prediction Networks](https://aclanthology.org/2021.acl-long.492/) | [官方](https://github.com/HangYang-NLP/DE-PPN)、[非官方](https://github.com/Spico197/DE-PPN) | [ChFinAnn](https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip) |
| 2021 | arxiv(待发表) | (PTPCG)[Efficient Document-level Event Extraction via Pseudo-Trigger-aware Pruned Complete Graph](https://arxiv.org/abs/2112.06013) |          [Code](https://github.com/Spico197/DocEE)           | [ChFinAnn](https://github.com/dolphin-zs/Doc2EDAG/blob/master/Data.zip) |

### 开放域事件抽取

| 年份 | 来源 |                             名称                             |                        源码                         |                            数据集                            |
| :--: | :--: | :----------------------------------------------------------: | :-------------------------------------------------: | :----------------------------------------------------------: |
| 2019 | ACL  | [Open Domain Event Extraction Using Neural Latent Variable Models](https://aclanthology.org/P19-1276/) | [Code](https://github.com/lx865712528/ACL2019-ODEE) | [Data](https://drive.google.com/file/d/1KjL3mAxj9nmzqC75s2rNaT6x6CJBZZTj/view) |
| 2022 | TOIS | [A Multi-Channel Hierarchical Graph Attention Network for Open Event Extraction](https://dl.acm.org/doi/10.1145/3528668) |     [Code](https://github.com/hawisdom/DL-OEE)      |                     CoNLL-2009 & ACE2005                     |

## 基础阅读
| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2019 |  | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) |  |  |
| 2019 | | ELG: An Event Logic Graph(提出事理图谱) | | |
| 2018 | | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://readpaper.com/paper/2963341956) | | |
| 2017 | | [Attention is All you Need](https://readpaper.com/paper/2963403868) | | |

## 扩展阅读

### 综述

| 年份 | 来源 | 名称 | 源码 | 数据集 |
| :-----: | :-----:| :-----:| :-----: | :-----: |
| 2021 | IEEE Access | [A Survey on Event Extraction for Natural Language Understanding: Riding the Biomedical Literature Wave](https://ieeexplore.ieee.org/document/9627684) |  |  |
| 2021 | IEEE DSC | [Survey on social event detection](https://ieeexplore.ieee.org/document/9750511) |  |  |
| 2021 | [LNNS](https://link.springer.com/bookseries/15179) | [Deep Learning Approaches to Detect Real Time Events Recognition in Smart Manufacturing Systems – A Short Survey](https://link.springer.com/chapter/10.1007/978-3-030-84910-8_20) |  |  |
| 2021 |  TKDE 2022 | [**What is Event Knowledge Graph: A Survey**](https://arxiv.org/abs/2112.15280) |  |  |
| 2020 | AI Open | [Extracting Events and Their Relations from Texts: A Survey on Recent Research Progress and Challenges](https://www.sciencedirect.com/science/article/pii/S266665102100005X?via%3Dihub) |  |  |
| 2020 | [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems) | [A survey on multi-modal social event detection](https://www.sciencedirect.com/science/article/pii/S0950705120301271?via%3Dihub) |  |  |
| 2020 | CCKS | [A Survey on Event Relation Identification](https://link.springer.com/chapter/10.1007/978-981-16-1964-9_14) |  |  |
| 2019 | IEEE Access | [A Survey of Event Extraction From Text](https://ieeexplore.ieee.org/document/8918013) |  |  |
| 2019 | 计算机科学 | [元事件抽取研究综述](https://www.jsjkx.com/CN/10.11896/j.issn.1002-137X.2019.08.002) |  |  |
| 2019 | ACM Trans | [How Deep Features Have Improved Event Recognition in Multimedia: A Survey](https://dl.acm.org/doi/10.1145/3306240) |  |  |
| 2018 | IJCAI | [Event Coreference Resolution: A Survey of Two Decades of Research](https://www.ijcai.org/proceedings/2018/773) |  |  |
| 2020 |  | [Reviews on Event Knowledge Graph Construction Techniques and Application-ReadPaper](http://www.c-a-m.org.cn/EN/10.3969/j.issn.1006-2475.2020.01.003) | | |
| 2021 | KSEM | [Event Relation Reasoning Based on Event Knowledge Graph](https://link.springer.com/chapter/10.1007/978-3-030-82136-4_40) | | |
| 2020 |  | [Introduction: What Is a Knowledge Graph?](https://link.springer.com/content/pdf/10.1007%2F978-3-030-37439-6_1.pdf) | | |
| 2021 |  | [OEKG - The Open Event Knowledge Graph](http://ceur-ws.org/Vol-2829/paper5.pdf) | | [Data](https://oekg.l3s.uni-hannover.de/data) |
| 2021 | CCKS | [MEED: A Multimodal Event Extraction Dataset](https://link.springer.com/chapter/10.1007/978-981-16-6471-7_23) |      | |

### 事理图谱

|   年份    |         来源         |                             名称                             |                             源码                             | 数据集 |
| :-------: | :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----: |
|   2020    | 计算机与现代化(CNKI) | [事件知识图谱构建技术与应用综述](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2020&filename=JYXH202001005&uniplatform=NZKPT&v=R-yruosM4vT_LhO2b-Qn2PzbUY-rgOez3a94GSCtf17lec6pZdk5UoP-I70xfYcc) |                                                              |        |
|   2021    |  大数据（中文期刊）  | [事件图谱的构建、推理与应用 ](http://www.infocomm-journal.com/bdr/CN/abstract/abstract171352.shtml) |                                                              |        |
|   2022    |         ACL          | [MMEKG: Multi-modal Event Knowledge Graph towards Universal Representation across Modalities](https://aclanthology.org/2022.acl-demo.23/) |                                                              |        |
|   2020    |        哈工大        | [面向金融领域的事理图谱构建关键技术研究](https://chn.oversea.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1020400011.nh&uniplatform=OVERSEA&v=t3vgLCmLnsHff3yx2H7k3wqmMvALzLmpG1StrbVB5HTy-h9T3hKVRrU_3q67BlFd) |                                                              |        |
| 2018-2021 |        IJCAI         | [SGNN 哈工大刘挺的博士 面向文本事件预测的事理图谱构建及应用方法研究_李忠阳](https://t.cnki.net/kcms/detail?v=sLRZPqxRYE3pHnscegK63uj1X-Ak4AimBdeP_sQplqNY172D9MXuhmfbybsKxfLKTUk3oB2bEU_16Ldfn1Or7zlQUDWyCcTKzPQoTeY3ZC3-PfN-WFhfpBSZ7MWX-ZguAr9UppBkGYI=&uniplatform=NZKPT) | [eecrazy/ConstructingNEEG_IJCAI_2018](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018) |        |

## 论文查找及管理工具

- [dblp: computer science bibliography](https://dblp.org/)
- [arXiv.org e-Print archive](https://arxiv.org/)
- [Sci-Hub](https://sci-hub.se/)
- [ReadPaper](https://readpaper.com/)
- [zotero](https://www.zotero.org/)

## GNN相关
+ B站课程：
    + [李沐的零基础多图详解图神经网络](https://www.bilibili.com/video/BV1iT4y1d7zP?spm_id_from=333.880.my_history.page.click&vd_source=137a8d9e49a8aecb804950139f2cd561)
    + [图卷积神经网络（GCN）的数学原理详解](https://www.bilibili.com/video/BV1Vw411R7Fj?p=1&vd_source=137a8d9e49a8aecb804950139f2cd561)
+ Distill上的两篇相关技术博客
    + [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)  P.S. 这篇博客就是李沐那个视频里讲的博客
    + [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)
+ 工具框架
  + [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#)
  + [Deep Graph Library](https://www.dgl.ai/)
  + [jraph](https://github.com/deepmind/jraph)
  + [Spektral](https://graphneural.network/)
+ 仓库
  + [GNNs-Recipe](https://github.com/dair-ai/GNNs-Recipe): 这个仓库整理了很多关于GNN方面的论文、代码、入门知识、工具等。

## 数据集
> 基本上所有事件时序关系的数据集在标注时都会遵循[TimeML](http://timeml.org/site/publications/timeMLdocs/timeml_1.2.1.html)规范。

|数据集名称| 年份 | 来源 | 论文名称 | 描述 | 下载 |
| :-----: | :-----: | :-----:| :-----:| :-----:| :-----:|
| ConceptNet | 2017 | | [ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://readpaper.com/paper/2561529111) |a large-scale commonsense knowledge graph for commonsense concepts, entities, events and relations||
| MATRES | 2018 | | [A Multi-Axis Annotation Scheme for Event Temporal Relations](https://aclanthology.org/P18-1122/) |a new benchmark dataset for TempRel extraction, which is developed from TempEval3 (UzZaman et al., 2013). It annotates on top of 275 documents with TempRels BEFORE, AFTER, EQUAL, and VAGUE. Particularly, the annotation process of MATRES has defined four axes for the actions of events, i.e. main, intention, opinion, andhypothetical axes. The TempRels are considered for all event pairs on the same axis and within a context of two adjacent sentences. The labels are decided by comparing the starting points of the events. The multi-axis annotation helped MATRES to achieve a high IAA of 0.84 in Cohen's Kappa.||
| HiEve |  | |  |a news corpus that contains 100 articles. Within each article, annotations are given for both subevent and coreference relations. The HiEve adopted the IAA measurement proposed for TempRels by (UzZaman and Allen, 2011), resulting in 0.69 F1.||
| TempEval3 | 2013 | | [Evaluating Time Expressions, Events, and Temporal Relations](https://aclanthology.org/S13-2001.pdf) |||
| RED | 2016 | | [Richer Event Description: Integrating event coreference with temporal, causal and bridging annotation]([Richer Event Description: Integrating event coreference with temporal, causal and bridging annotation-ReadPaper](https://readpaper.com/paper/2561222820)) |contains 35 news articles with annotations for event complexes that contain both membership relations and TempRels.||
| ESTER | 2021 | | [ESTER: A Machine Reading Comprehension Dataset for Event Semantic Relation Reasoning](https://readpaper.com/paper/3153067519) |a comprehensive machine reading comprehension (MRC) dataset for Event Semantic Relation Reasoning. The dataset leverages natural language queries to reason about the five most common event semantic relations, provides more than 6K questions and captures 10.1K event relation pairs.||
| CausalBank |  | |  |大规模英文因果数据集||
| TimeBank |  | |  |仅仅标注了部分容易识别的关系子集||
| TimeBank-Dense | 2014 |  | [An Annotation Framework for Dense Event Ordering](https://aclanthology.org/P14-2082/) |BEFORE , AFTER , INCLUDES , IS INCLUDED, SIMULTANEOUS , VAGUE||
| GDELT事件库 |  | |  |利用超过100 种语言的全球新闻媒体数据, 自动发现并记录了自 1979 年 1 月 1 日以来的所有人类社会主要事件|[Site](https://www.gdeltproject.org/)|
| LinkedData |  | |  |150年的新闻文章和200亿条关系的知识图谱||
| VLEP |  | |  |一个视频与字幕结合的多模态事件预测数据集||
| EventKG | | | ||[Data](https://oekg.l3s.uni-hannover.de/data)|
| OEKG | | | ||[Data](https://oekg.l3s.uni-hannover.de/data)|
| 中文突发事件语料库（CEC）||||中文突发事件语料库是由上海大学（语义智能实验室）所构建。根据国务院颁布的《国家突发公共事件总体应急预案》的分类体系，从互联网上收集了5类（地震、火灾、交通事故、恐怖袭击和食物中毒）突发事件的新闻报道作为生语料，然后再对生语料进行文本预处理、文本分析、事件标注以及一致性检查等处理，最后将标注结果保存到语料库中，CEC合计332篇。|[CEC-Corpus](https://github.com/shijiebei2009/CEC-Corpus)|
|  | | | [事件抽取相关数据集整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/443260886) |||
|  | | | [开放资源：面向事件时序因果关系识别的17类开源标注数据集总结 - 墨天轮 (modb.pro)](https://www.modb.pro/db/379577) |||

