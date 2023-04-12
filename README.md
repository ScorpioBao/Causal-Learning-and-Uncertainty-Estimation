# Cusal-Learning-and-uncertainty-estimation
该站点用于整理本人研究生期间相关研究方向的学习资源，包括论文、代码、数据集和博客等学习资源等。
### 目录
#### 一、论文
1. [因果推理](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning)
	- [1.1因果效应评估（Causal Effect Estimation）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning/Causal%20Effect%20Estmation))
	- [1.2因果发现（Causal Discovery）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning/Causal%20Discovery)
2. [因果表征学习](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Representation%20Learning)
	- [2.1分布外泛化（Out-of-Distribution Generalization）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Representation%20Learning/Out-of-Distribution%20Generalization)
	- [2.2稳定学习（Stable Learning）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Representation%20Learning/Stable%20Learning)
3. [不确定性估计](#3不确定性估计)
	- 贝叶斯方法（Bayesian）
	- 集成方法（Ensemble）
	- 证据深度学习（Evidential Deep Learning)
4. [ 不确定性估计在不同领域的应用](#4、不确定性估计在不同领域的应用)
5. 深度学习模型校准的相关工作
#### 二、[代码和数据集](#二、代码和数据集)
#### 三、[博客](#三、博客)

# 一、论文
### 1 因果推理(相关论文可以通过目录中的链接访问)
### 2 因果表征学习(相关论文可以通过目录中的链接访问)
### 3不确定性估计
#### （1）综述
- Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt, M., Feng, J., ... & Zhu, X. X. (2021). A survey of uncertainty in deep neural networks. _arXiv preprint arXiv:2107.03342_.
- Zhou, T., Han, T., & Droguett, E. L. (2022). Towards trustworthy machine fault diagnosis: A probabilistic Bayesian deep learning framework. _Reliability Engineering & System Safety_, _224_, 108525.
- Abdar, M., Pourpanah, F., Hussain, S., Rezazadegan, D., Liu, L., Ghavamzadeh, M., ... & Nahavandi, S. (2021). A review of uncertainty quantification in deep learning: Techniques, applications and challenges. _Information Fusion_, _76_, 243-297.
- Uncertainty in Deep Learning（Gal博士论文）
- He, W., & Jiang, Z. (2023). A Survey on Uncertainty Quantification Methods for Deep Neural Networks: An Uncertainty Source Perspective. _arXiv preprint arXiv:2302.13425_.

#### （2）贝叶斯方法：
- Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In _international conference on machine learning_ (pp. 1050-1059). PMLR.（将Dropout看做贝叶斯近似的经典论文）
- Harakeh, A., Smart, M., & Waslander, S. L. (2020, May). Bayesod: A bayesian approach for uncertainty estimation in deep object detectors. In _2020 IEEE International Conference on Robotics and Automation (ICRA)_ (pp. 87-93). IEEE.（目标检测中的贝叶斯不确定性估计）
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. _Advances in neural information processing systems_, _30_.（不确定性估计必读论文，将不确定性分为数据不确定性以及模型不确定性，并介绍了在分类和回归中不确定性估计的建模方法）
- Louizos, C., & Welling, M. (2017, July). Multiplicative normalizing flows for variational bayesian neural networks. In _International Conference on Machine Learning_ (pp. 2218-2227). PMLR.（变分贝叶斯神经网络，EDL论文中的对比方法）

#### （3）集成方法
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. _Advances in neural information processing systems_, _30_.（集成方法的开山之作）
#### （4）证据深度学习
- Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. _Advances in neural information processing systems_, _31_.（证据分类）
- Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential regression. _Advances in Neural Information Processing Systems_, _33_, 14927-14937.（证据回归）
- Ulmer, D. (2021). A survey on evidential deep learning for single-pass uncertainty estimation. _arXiv preprint arXiv:2110.03051_.（综述）
### 4、不确定性估计在不同领域的应用
#### （1）分割
- Kwon, Y., Won, J. H., Kim, B. J., & Paik, M. C. (2020). Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation. _Computational Statistics & Data Analysis_, _142_, 106816.（贝叶斯不确定性）
- Li, H., Nan, Y., Del Ser, J., & Yang, G. (2022). Region-based evidential deep learning to quantify uncertainty and improve robustness of brain tumor segmentation. _Neural Computing and Applications_, 1-15.（证据不确定性）
#### （2）开集识别
- Bao, W., Yu, Q., & Kong, Y. (2021). Evidential deep learning for open set action recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ (pp. 13349-13358).（开集识别）
- Corbière, C., Lafon, M., Thome, N., Cord, M., & Pérez, P. (2021, September). Beyond First-Order Uncertainty Estimation with Evidential Models for Open-World Recognition. In _ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning_.（正则化项）
#### （3）域自适应
- Chen, L., Lou, Y., He, J., Bai, T., & Deng, M. (2022, June). Evidential neighborhood contrastive learning for universal domain adaptation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 36, No. 6, pp. 6258-6267).
#### （4）多视图学习
- Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2022). Trusted multi-view classification with dynamic evidential fusion. _IEEE transactions on pattern analysis and machine intelligence_.（证据多视图分类）
- Ma, H., Han, Z., Zhang, C., Fu, H., Zhou, J. T., & Hu, Q. (2021). Trustworthy multimodal regression with mixture of normal-inverse gamma distributions. _Advances in Neural Information Processing Systems_, _34_, 6881-6893.（证据多模态回归）





### 二、代码和数据集
相关论文的代码以及数据集可以在[Paper With Code ](https://paperswithcode.com/)搜索获取，如果Paper With Code 中没有收录，可直接在GitHub输入论文关键字搜索相关代码
### 三、博客
##### - 因果推理

##### - 因果表征学习

##### - 不确定性估计





<br>


