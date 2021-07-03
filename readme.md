
************************************************************
* OATES -- Online Multi-Stream Transfer Effort Estimator   *
************************************************************

This repository contains OATES code used in the following paper:

MINKU, L.L. Multi-Stream Online Transfer Learning For Software Effort Estimation -- Is It Necessary? International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE) 2021.

It is released under GNU General Public License version 3.0 (GPLv3). It also makes use of the third party MOA Release 2016.04, whose code and dependencies are available in moa-2016.04-sources.jar. The license for MOA Release 2016.04 is GNU General Public License version 3.0 (GPLv3). The licenses of its dependencies can be found within the moa-2016.04-sources.jar file.

To use OATES implementation, you need to use the following classes together with MOA-2016.04's code:

- interface classifiers.meta.MappingFunction
- classifiers.meta.SimpleLinearMappingFunction
- classifiers.meta.SimpleLinearMappingFunctionTest
- classifiers.meta.OATES
- classifiers.meta.OATESTest
- clusterers.threshold.ProductivitySplitClusterer
- clusterers.threshold.ProductivitySplitClustererTest
- cluster.ProductivityThresholdCluster

To use WEKA's classifiers in OATES as done in the paper, use:

- classifiers.meta.WEKAClassifierTrainSlidingWindow
- classifiers.meta.WEKAClassifierTrainSlidingWindowTest
- classifiers.meta.WEKALogClassifierTrainSlidingWindow

The MOA-2016.04 classes below were updated to report mean absolute error of the regression predictions in the log scale. The updates are annotated with comments "<---le". The following classes from the OATES repository need to be used to replace the corresponding files in the MOA-2016.04 code:

- moa.evaluation.WindowRegressionPerformanceEvaluator
- moa.evaluation.BasicRegressionPerformanceEvaluator

