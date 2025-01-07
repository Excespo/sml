在SML课程的期末，您需要提交一份关于自己选择的最终项目的4页报告。报告应包含至少以下几个部分：

1. **5分**：您选择的工程主题的描述；主题可以自由选择，但应与工程问题相关，而非纯粹的计算机科学话题，如图像、语音或视觉。
On the prediction of critical heat flux ... paper 
2. **10分**：描述相关的技术问题及其困难。
- Mentioned in the paper
- Additional data from kaggle with missing values, need to be preprocessed
- Docker environment for running the code (Build from scratch)
- Code implementation because the paper doesn't provide its official code
- 
3. **10分**：回顾与您的问题相关的已发布文献和现有方法。
References in papar +
- Rule based/Statistical methods, lookup table/liu's model 2.1.1.1+2.1.1.2
- Pure ML methods with neural network and random forest 2.1.2
- Combination of the two

4. **20分**：提出并描述您选择的解决问题的算法。这应该是一个已发布的方法或已验证的技术，以避免最后的惊喜。最好选择课程中覆盖的方法，如支持向量机、随机森林、深度神经网络、物理信息神经网络等。
Mentioned in Q3

5. **20分**：使用您选择的语言和框架实现方法。您可以使用TensorFlow或PyTorch。复用现有代码是可以的，但必须引用您的来源，并描述您对代码所做的更改和补充。提交您的代码与报告一起。这不计入报告的篇幅。
Pytorch + Manual implementation

6. **5分**：描述您采取的验证代码正确性的方法。
Compare with results in paper

7. **10分**：选择并描述您用于代码验证和基准测试的数据集、问题设置和参数。
- Real data from paper https://www.kaggle.com/datasets/saurabhshahane/predicting-heat-flux
- Synthetic data from kaggle https://www.kaggle.com/competitions/playground-series-s3e15/data

Real data is ready for use, synthetic data needs to be preprocessed (Mentioned in Q2)

Train/test split from the complete dataset in ratio of 8 - 2 with random sampling

Benchmark: (Metric)
- RMSE(root mean square error) (pred - ground truth)
- rRMSE(relative root mean square error) ((pred - ground truth) / ground truth)
- Cumulative distribution function (CDF) x = rRMSE, y = proportion of data with rRMSE < x

Problem: 
Prediction of critical heat flux (CHF)

Parameters:
- Neural Network Architecture: 
    8->16->16->1, 8->32->32->32->16->24->32->1
- activation function: ReLU
- loss function: MSE (Mean Squared Error)
- optimizer: SGD (Stochastic Gradient Descent)(use for back propagation)
- learning rate: 1e-3
- batch size: 1 (make sure the model learns every sample)
- epochs: 2000 


8. **15分**：绘制并展示基准测试结果。讨论并解释结果。确保探索多种输入和算法参数，以全面了解方法的表现。

- Comparison between models (Pure physical models, pure ML models, combination of the two->Hybrid model) (Needs plots)
- Distribution of ground truth (model's initialization, single data distribution should follow Central Limit Theorem, multi-dim features should follow multivariate normal distribution, and the model map this distribution to the output's distribution) (Needs plots)
- Batch size (affect learning from outliers) (Needs plots)
- Data Mixture of real data and synthetic data (syn data quality so low -> 这部分合成数据是kaggle伪造用来拟合x-e-out(feature之一, equilibrium quality)的, 由于大量的缺失值需要补全,我们用KNN聚类来补全缺失值,引入了大量的数据扰动,扰动后的数据给真实值的学习造成了很大的难度,导致模型loss很难下降, 最后我们只能选择弃用这部分数据)

9. **5分**：讨论所提出方法的局限性并提出改进意见。

ref to paper

我们建议保持项目简单，设定切实可行的目标。如果一切顺利，可以添加“额外”结果。最好从简单的项目开始，逐步增加内容，而不是一开始就设定过于雄心勃勃的目标，然后在最后放弃，改为更简单的内容。

为了帮助您提前规划并确保按时提交，要求您提交一个项目计划。截止日期为1月4日。该计划占10分。计划应为一页纸，包含：

- 项目简介
- 相关方法的简要文献调查
- 任务时间表，包含从现在到项目成功提交的日期。

提交说明：项目有三个作业：

- 初步计划：10分
- 最终项目代码（上传一个文件，如Python笔记本或Python代码）：20分
- 最终项目报告（上传一个PDF文件）：80分

如果您希望，可以与最多1个同学组成2人小组进行项目。无论是否是小组项目，您投入的工作量应保持一致。

对于2人小组，报告页面数为6页。

除了提交小组报告外，请确保在PDF报告的开头列出所有小组成员。每个小组成员还必须提交一份简短的文档，描述每个成员（包括您自己）在项目中的贡献。