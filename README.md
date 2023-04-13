# Cusal-Learning-and-uncertainty-estimation
该站点整理了”因果推理与不确定性估计“相关研究方向的论文、代码、博客等学习资源。
# 目录
#### 一、论文
- [因果推理](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning)
	- [因果效应评估（Causal Effect Estimation）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning/Causal%20Effect%20Estmation))
	- [因果发现（Causal Discovery）](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Learning/Causal%20Discovery)
- [因果表征学习](#2因果表征学习)
	- [综述](#1综述)
	- [分布外泛化（Out-of-Distribution Generalization）](#2分布外泛化)
	- [稳定学习（Stable Learning）](#3稳定学习)
	- [消除偏差（Debias）](#4消除偏差)
-  [不确定性估计](#3不确定性估计)
	- [综述](#1综述)
	- [贝叶斯方法（Bayesian）](#2贝叶斯方法)
	- [集成方法（Ensemble）](#3集成方法)
	- [证据深度学习（Evidential Deep Learning)](#4证据深度学习)
- [ 不确定性估计在不同领域的应用](#4不确定性估计在不同领域的应用)
	- [分割](#1分割)
	- [目标检测](#2目标检测)
	- [开集识别](#3开集识别)
	- [分布外泛化](#4分布外泛化)
	- [多视图学习](#5多视图学习)
- [深度学习模型校准的相关工作](#5深度学习模型校准的相关工作)
- [代码和数据集](#二代码和数据集)
- [博客](#三博客)
	- [因果推理](#1因果推理)
	- [不确定性估计](#2不确定性估计)
- [交流](#四交流)
# 一、论文
### 1、 因果推理(相关论文可以通过目录中的链接访问)
### 2、 因果表征学习
#### （1）综述
- Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward causal representation learning. _Proceedings of the IEEE_, _109_(5), 612-634.（因果表征学习综述）
- Lu, C., Wu, Y., Hernández-Lobato, J. M., & Schölkopf, B. (2021). Invariant causal representation learning for out-of-distribution generalization. In _International Conference on Learning Representations_.（不变因果表征学习）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （2）分布外泛化
*相关论文可以在[分布外泛化](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Representation%20Learning/Out-of-Distribution%20Generalization)文件夹下查看*
- Xu, R., Zhang, X., Shen, Z., Zhang, T., & Cui, P. (2022, June). A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization. In _International Conference on Machine Learning_ (pp. 24803-24829). PMLR.（协变量偏移）
- Liu, C., Sun, X., Wang, J., Tang, H., Li, T., Qin, T., ... & Liu, T. Y. (2021). Learning causal semantic representation for out-of-distribution prediction. _Advances in Neural Information Processing Systems_, _34_, 6155-6170.（因果语义表示学习）
- Zhang, X., Xu, Z., Xu, R., Liu, J., Cui, P., Wan, W., ... & Li, C. (2022). Towards domain generalization in object detection. _arXiv preprint arXiv:2203.14387_.（目标检测中的域泛化）
- Shen, Z., Liu, J., He, Y., Zhang, X., Xu, R., Yu, H., & Cui, P. (2021). Towards out-of-distribution generalization: A survey. _arXiv preprint arXiv:2108.13624_.（综述）
- Li, X., Dai, Y., Ge, Y., Liu, J., Shan, Y., & Duan, L. Y. (2022). Uncertainty modeling for out-of-distribution generalization. _arXiv preprint arXiv:2202.03958_.（OODG不确定性建模）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （3）稳定学习
*相关论文可以在[稳定学习](https://github.com/ScorpioBao/Causal-Learning-and-Uncertainty-Estimation/tree/master/Causal%20Representation%20Learning/Stable%20Learning)文件夹下查看*
- Zhang, X., Cui, P., Xu, R., Zhou, L., He, Y., & Shen, Z. (2021). Deep stable learning for out-of-distribution generalization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ (pp. 5372-5382).（StableNet）
- Liu, J., Hu, Z., Cui, P., Li, B., & Shen, Z. (2021, July). Heterogeneous risk minimization. In _International Conference on Machine Learning_ (pp. 6804-6814). PMLR.（异质风险最小化）
- Cui, P., & Athey, S. (2022). Stable learning establishes some common ground between causal inference and machine learning. _Nature Machine Intelligence_, _4_(2), 110-115.（稳定学习与因果推断和机器学习之间的共同点）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （4）消除偏差
- Wang, T., Zhou, C., Sun, Q., & Zhang, H. (2021). Causal attention for unbiased visual recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ (pp. 3091-3100).（因果注意力）
- Niu, Y., Tang, K., Zhang, H., Lu, Z., Hua, X. S., & Wen, J. R. (2021). Counterfactual vqa: A cause-effect look at language bias. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ (pp. 12700-12710).（反事实VQA）
- Yang, X., Zhang, H., & Cai, J. (2021). Deconfounded image captioning: A causal retrospect. _IEEE Transactions on Pattern Analysis and Machine Intelligence_.（去除混淆偏差）
- Nam, J., Cha, H., Ahn, S., Lee, J., & Shin, J. (2020). Learning from failure: De-biasing classifier from biased classifier. _Advances in Neural Information Processing Systems_, _33_, 20673-20684.（从有偏分类器学习去偏分类器）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
### 3、不确定性估计
#### （1）综述
- Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt, M., Feng, J., ... & Zhu, X. X. (2021). A survey of uncertainty in deep neural networks. _arXiv preprint arXiv:2107.03342_.
- Abdar, M., Pourpanah, F., Hussain, S., Rezazadegan, D., Liu, L., Ghavamzadeh, M., ... & Nahavandi, S. (2021). A review of uncertainty quantification in deep learning: Techniques, applications and challenges. _Information Fusion_, _76_, 243-297.
- Uncertainty in Deep Learning（Gal博士论文）
- He, W., & Jiang, Z. (2023). A Survey on Uncertainty Quantification Methods for Deep Neural Networks: An Uncertainty Source Perspective. _arXiv preprint arXiv:2302.13425_.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. _Machine Learning_, _110_, 457-506.（数据和模型不确定性）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （2）贝叶斯方法：
- Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In _international conference on machine learning_ (pp. 1050-1059). PMLR.（将Dropout看做贝叶斯近似的经典论文）
- **Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. _Advances in neural information processing systems_, _30_.（不确定性估计必读论文，将不确定性分为数据不确定性以及模型不确定性，并介绍了在分类和回归中不确定性估计的建模方法）**
- Louizos, C., & Welling, M. (2017, July). Multiplicative normalizing flows for variational bayesian neural networks. In _International Conference on Machine Learning_ (pp. 2218-2227). PMLR.（变分贝叶斯神经网络，EDL论文中的对比方法）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （3）集成方法
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. _Advances in neural information processing systems_, _30_.（集成方法的开山之作）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （4）证据深度学习
- **Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. _Advances in neural information processing systems_, _31_.（证据分类）**
- **Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential regression. _Advances in Neural Information Processing Systems_, _33_, 14927-14937.（证据回归）**
- Sensoy, M., Kaplan, L., Cerutti, F., & Saleki, M. (2020, April). Uncertainty-aware deep classifiers using generative models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 34, No. 04, pp. 5620-5627).（EDL作者的另一篇论文）
- Malinin, A., & Gales, M. (2018). Predictive uncertainty estimation via prior networks. _Advances in neural information processing systems_, _31_.（同样使用狄利克雷分布建模不确定性的另一种方法）
- Ulmer, D. (2021). A survey on evidential deep learning for single-pass uncertainty estimation. _arXiv preprint arXiv:2110.03051_.（证据不确定性综述）
- Zhao, X., Ou, Y., Kaplan, L., Chen, F., & Cho, J. H. (2019). Quantifying classification uncertainty using regularized evidential neural networks. _arXiv preprint arXiv:1910.06864_.（证据基础上添加正则化项）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
### 4、不确定性估计在不同领域的应用
#### （1）分割
- Kwon, Y., Won, J. H., Kim, B. J., & Paik, M. C. (2020). Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation. _Computational Statistics & Data Analysis_, _142_, 106816.（贝叶斯不确定性）
- Li, H., Nan, Y., Del Ser, J., & Yang, G. (2022). Region-based evidential deep learning to quantify uncertainty and improve robustness of brain tumor segmentation. _Neural Computing and Applications_, 1-15.（证据不确定性）
- Zou, K., Yuan, X., Shen, X., Chen, Y., Wang, M., Goh, R. S. M., ... & Fu, H. (2023). EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap. _arXiv preprint arXiv:2301.00349_.（证据不确定性）
- Zhou, X., Yue, X., Xu, Z., Denoeux, T., & Chen, Y. (2021, December). Deep neural networks with prior evidence for bladder cancer staging. In _2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ (pp. 1221-1226). IEEE.（证据医学影像分割）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （2）目标检测
- Harakeh, A., Smart, M., & Waslander, S. L. (2020, May). Bayesod: A bayesian approach for uncertainty estimation in deep object detectors. In _2020 IEEE International Conference on Robotics and Automation (ICRA)_ (pp. 87-93). IEEE.（目标检测中的贝叶斯不确定性估计）
- Feng, D., Harakeh, A., Waslander, S. L., & Dietmayer, K. (2021). A review and comparative study on probabilistic object detection in autonomous driving. _IEEE Transactions on Intelligent Transportation Systems_, _23_(8), 9961-9980.（自动驾驶中的概率目标检测）
- Hang, Q., Li, Z., Dong, Y., & Yue, X. (2022, November). Uncertainty-Aware Deep Open-Set Object Detection. In _Rough Sets: International Joint Conference, IJCRS 2022, Suzhou, China, November 11–14, 2022, Proceedings_ (pp. 161-175). Cham: Springer Nature Switzerland.（证据目标检测）
- Miller, D. (2021). _Epistemic uncertainty estimation for object detection in open-set conditions_ (Doctoral dissertation, Queensland University of Technology).（开集目标检测）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （3）开集识别
- Bao, W., Yu, Q., & Kong, Y. (2021). Evidential deep learning for open set action recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ (pp. 13349-13358).（开集识别）
- Corbière, C., Lafon, M., Thome, N., Cord, M., & Pérez, P. (2021, September). Beyond First-Order Uncertainty Estimation with Evidential Models for Open-World Recognition. In _ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning_.（正则化项）
- Corbière, C., Lafon, M., Thome, N., Cord, M., & Pérez, P. (2021, September). Beyond First-Order Uncertainty Estimation with Evidential Models for Open-World Recognition. In _ICML 2021 Workshop on Uncertainty and Robustness in Deep Learning_.（证据用于开放世界识别）
- Mundt, M., Pliushch, I., Majumder, S., & Ramesh, V. (2019). Open set recognition through deep neural network uncertainty: Does out-of-distribution detection require generative classifiers?. In _Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops_.（OOD检测）
- Zhou, T., Han, T., & Droguett, E. L. (2022). Towards trustworthy machine fault diagnosis: A probabilistic Bayesian deep learning framework. _Reliability Engineering & System Safety_, _224_, 108525.（贝叶斯故障诊断）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （4）分布外泛化
- Chen, L., Lou, Y., He, J., Bai, T., & Deng, M. (2022, June). Evidential neighborhood contrastive learning for universal domain adaptation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 36, No. 6, pp. 6258-6267).（证据领域对比学习）
- Qiao, F., & Peng, X. (2021). Uncertainty-guided model generalization to unseen domains. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ (pp. 6790-6800).（不确定性指导的数据增广）
- Zhao, L., Liu, T., Peng, X., & Metaxas, D. (2020). Maximum-entropy adversarial data augmentation for improved generalization and robustness. _Advances in Neural Information Processing Systems_, _33_, 14435-14447.（对抗数据增广-最大熵）
- Li, X., Dai, Y., Ge, Y., Liu, J., Shan, Y., & Duan, L. Y. (2022). Uncertainty modeling for out-of-distribution generalization. _arXiv preprint arXiv:2202.03958_.（OODG的不确定性建模）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
#### （5）多视图学习
- Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2022). Trusted multi-view classification with dynamic evidential fusion. _IEEE transactions on pattern analysis and machine intelligence_.（证据多视图分类）
- Ma, H., Han, Z., Zhang, C., Fu, H., Zhou, J. T., & Hu, Q. (2021). Trustworthy multimodal regression with mixture of normal-inverse gamma distributions. _Advances in Neural Information Processing Systems_, _34_, 6881-6893.（证据多模态回归）
- Geng, Y., Han, Z., Zhang, C., & Hu, Q. (2021, May). Uncertainty-aware multi-view representation learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 35, No. 9, pp. 7545-7553).（多视图回归-数据不确定性建模）

👆 [<b>BACK to Table of Contents</b> -->](#目录)
### 5、深度学习模型校准的相关工作
- **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, July). On calibration of modern neural networks. In _International conference on machine learning_ (pp. 1321-1330). PMLR.（分类校准）**
- **Kuleshov, V., Fenner, N., & Ermon, S. (2018, July). Accurate uncertainties for deep learning using calibrated regression. In _International conference on machine learning_ (pp. 2796-2804). PMLR.（回归校准）**
- Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P., & Dokania, P. (2020). Calibrating deep neural networks using focal loss. _Advances in Neural Information Processing Systems_, _33_, 15288-15299.（Focal Loss 校准分类）
- Krishnan, R., & Tickoo, O. (2020). Improving model calibration with accuracy versus uncertainty optimization. _Advances in Neural Information Processing Systems_, _33_, 18237-18248.（考虑不确定性校准模型）
- Thulasidasan, S., Chennupati, G., Bilmes, J. A., Bhattacharya, T., & Michalak, S. (2019). On mixup training: Improved calibration and predictive uncertainty for deep neural networks. _Advances in Neural Information Processing Systems_, _32_.（mixup提高模型校准性能）

👆 [<b>BACK to Table of Contents</b> -->](#目录)

## 二、代码和数据集
相关论文的代码以及数据集可以在[Paper With Code ](https://paperswithcode.com/)搜索获取，如果Paper With Code 中没有收录，可直接在GitHub输入论文关键字搜索相关代码

👆 [<b>BACK to Table of Contents</b> -->](#目录)
## 三、博客
### 1、因果推理
- [e-CARE: 可解释的因果推理评测 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650805534&idx=1&sn=ecf1c78e642f46daa10c1217c3bd320d&chksm=8cb880f5bbcf09e3d060f3b96fb24c42ee60c52a1b10914e3af637e8f07c9c88c5ca6f02d17a#rd)
- [【Valse - 崔鹏】Out-of-Distribution 分布外泛化 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/419346109)
- [崔鹏团队：万字长文梳理「稳定学习」全景图 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/535602186)
- [【综述】离群/异常/新类检测？开集识别？分布外检测？一文搞懂其间异同！ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/426521773)
-  [因果表征学习最新综述：连接因果科学和机器学习的桥梁 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/355009051)
- [因果推断：因果表征学习的CV落地 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/400043237)

👆 [<b>BACK to Table of Contents</b> -->](#目录)
### 2、不确定性估计
- [(184条消息) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? 计算机视觉用于贝叶斯深度学习的不确定性_Xieyuanli_Chen的博客-CSDN博客](https://blog.csdn.net/weixin_39779106/article/details/78968982#1%E5%B0%86%E5%BC%82%E6%96%B9%E5%B7%AE%E5%81%B6%E7%84%B6%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E5%92%8C%E8%AE%A4%E7%9F%A5%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%9B%B8%E7%BB%93%E5%90%88%5D(https://blog.csdn.net/weixin_39779106/article/details/78968982#1%E5%B0%86%E5%BC%82%E6%96%B9%E5%B7%AE%E5%81%B6%E7%84%B6%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E5%92%8C%E8%AE%A4%E7%9F%A5%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%9B%B8%E7%BB%93%E5%90%88))
- [Bayesian inference problem, MCMC and variational inference | by Joseph Rocca | Towards Data Science](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29)
- [Uncertainty Estimation in CV - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/166617220)
- [What my deep model doesn't know... | Yarin Gal - Blog | Oxford Machine Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html)
- [Uncertainty in Deep Learning. How To Measure? | Towards Data Science](https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b)
- https://www.bilibili.com/video/BV1RJ411D7QA/
- [如何创造可信任的机器学习模型？先要理解不确定性 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755237&idx=3&sn=55beb3edcef0bb4ded4b56e1379efbda&chksm=871a94dbb06d1dcddc49272f77899561c0da5760f2dc6cfebd3877272a959e01c69105a8bac2#rd)
- [从最大似然到EM算法：一致的理解方式 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/5239)
- [ICML高引模型校准论文，一个好的工作是怎样的 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/323959089)

👆 [<b>BACK to Table of Contents</b> -->](#目录)
## 四、交流
- Email: senlinbao@gmail.com





<br>


