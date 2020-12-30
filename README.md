低光照重建

所提算法由两个网络组成，一个网络用于重构原始图像，一个网络用于暗光增强，两个网络共享编码层。


1.	基于语义和跨尺度注意力的深度网络去雾模型
传统的去雾模型基于合成的有雾和无雾图像学习得到映射关系，然而合成数据集是应用简化的物理模型合成数据，该模型仅考虑均匀的大气光和均匀的散射系数，因此无法适应实际的场景。而且在距离较远的地方雾浓度较大，导致远处的场景信息完全丢失，无法应用简单的模型恢复其真实场景。为了克服这一个问题，本研究提出了一种端到端的去雾模型，该模型利用估计的语义信息和纹理信息构造最终的去雾结果。浅层网络层的特征有助于恢复清晰图像的纹理信息，然而并非所有的特征都有助于恢复清晰图像。本研究提出了跨尺度注意力机制用于选择有效的浅层特征用于恢复最终的清晰图像。应用语义信息有助于提高去雾结果的清晰度和恢复由于雾而导致的场景丢失，通过注意力机制可以选择更加有效的浅层特征用于恢复纹理信息。因此本研究所提模型能够恢复清晰且纹理丰富的去雾结果。

2.	基于半监督的暗光增强算法
现有主流的暗光增强算法都是基于全监督的，这些算法通过大量的合成数据集训练得到暗光图像和增强图像之间的映射关系。虽然这些算法在合成数据集上面都取得了不错的增强效果，但是这些算法在实际暗光场景中的增强效果都不是很好。为了解决这个问题，本研究提出了一个半监督的暗光增强算法提高算法在自然暗光场景的泛化能力。所提算法基于暗光图像和增强图像是统一场景的不同表示。应用编码网络对自然暗光图像和合成暗光图像进行表示，然后通过重构网络对编码进行重构得到原始的输入图像，通过增强网络得到增强图像。通过重构网络可以学习得到自然暗光图像和合成暗光图像的通用表示，解决了合成暗光图像和自然暗光图像之间存在的差异。

3.	基于网络架构自动搜索的超分方法
设计一个可以用于图像超分的网络是非常复杂的。设计一个可以处理多种情况的超分网络是一个更加困难的任务。为了解决这个问题，本研究引入了网络架构自动搜索技术用于设计一个高效的超分网络。为了使网络能够利用时空信息，本研究还引入了3D卷积神经网络对得到的连续特征进行融合得到最终的超分结果。




拟解决的关键科学问题
（1）	如何应用跨尺度注意力和语义信息提高模型去雾能力
传统的去雾算法应用简化的大气散射模型进行去雾，然而当雾浓度较大时，容易导致局部区域场景完全丢失，无法应用大气散射模型恢复清晰图像。现有的研究表明，卷积神经网络模型可以从一个语义层恢复得到一个合理的清晰图像。因此可以通过语义信息恢复由于雾的作用而丢失的场景信息。语义信息还可以用于限制去雾解空间的复杂程度，有助于降低模型的复杂程度。由此可见，语义信息有助于提高模型的表达能力和去雾能力。
浅层网络包含大量的纹理、颜色和边界特征，然后这些特征通常比较冗余，为了使重要的浅层能够传递到深层网络，本研究提出了一个跨尺度注意力机制，该机制利用深层特征信息去选择对于深层特征更加互补的浅层特征。因此跨尺度的注意力机制有助于网络对浅层特征的选择，更有助于去雾结果中细节和纹理信息的恢复。因此，如何利用语义信息指导网络如何去恢复有雾图像和利用跨尺度注意力去选择合适的浅层特征恢复纹理和边缘是本研究要解决的关键问题之一。



（2）	如何利用半监督解决网络模型的泛化能力
目前基于深度学习的算法都是在大量合成数据集上面训练，虽然此类算法在合成数据集上取得了不错的效果。然而此类方法忽略了自然暗光图像和合成暗光图像之间的差异，因此此类方法在自然暗光图像上面可能增强效果不佳。为了解决这个问题，本项目应用半监督方法提高模型在自然暗光图像上的增强效果。半监督方法通过对自然暗光图像进行学习，可以减少模型在自然暗光图像上面的不适应性。因此利用半监督学习解决模型的泛化能力是本研究所需要解决的关键问题之一。
（3）	如何利用自动网络架构搜索和3D卷积神经网络构造高效的视频超分模型
目前的图像超分方法主要集中在空间超分辨率上面，对于视频中存在的帧与帧之间的关系研究和利用比较少，更没有考虑在实际应用中存在的雾和暗光等场景。首先，设计一个用于处理复杂场景的超分网络是极其复杂的。其次，如果直接对单帧视频进行超分，则容易导致超分结果存在不连续性。因此结合网络架构自动化搜索，结合交通监控视频中的时空信息，研究复杂环境下的视频超分辨率方法是本研究需要解决的另一个关键问题。



Zhengxiong Luo, Yan Huang, Shang Li, Liang Wang, Tieniu Tan. Unfolding the Alternating Optimization for Blind Super Resolution. NeurIPS, 2020.

图像超分：
早期的图像超分方法通过


Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew P. Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 2017.

Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In CVPRW, 2017.

Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image super-resolution. In CVPR, 2018.
M. Haris, G. Shakhnarovich, and N. Ukita, “Deep backprojection networks for super-resolution,” in CVPR, 2018.
Z. Li, J. Yang, Z. Liu, X. Yang, G. Jeon, and W. Wu. Feedback network for image super-resolution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3867–3876, 2019. 6

H. Ren, M. El-Khamy, and J. Lee, “Image super resolution based on fusing multiple convolution neural networks,” in CVPRW, 2017. 
Y. Hu, X. Gao, J. Li, Y. Huang, and H. Wang, “Single image superresolution via cascaded multi-scale cross network,” arXiv, 2018.
 Z. Hui, X. Wang, and X. Gao, “Fast and accurate single image superresolution via information distillation network,” in CVPR, 2018.

Ben Niu, Weilei Wen, Wenqi Ren, Xiangde Zhang, Lianping Yang, Shuzhen Wang, Kaihao Zhang, Xiaochun Cao, Haifeng Shen. Single Image Super-Resolution via a Holistic Attention Network. ECCV 2020.

SRResNet【Ledig】 和 EDSR【Lim】 通过残差模块提高超分模型的效果。SRDenseNet 则通过应用密集链接模块提高超分模型的效果。Zhang等人则通过结合全局残差和密集链接来提高深度模型的超分效果。DDBPN【Haris】 则通过反馈机制提高超分精度。Li等人则应用反馈机制提高模型的特征表达能力。为了得到不同尺度的特征表达，多个分支的网络被提出来并应用于超分模型的设计。Niu等人则通过通道和空间注意力机制提高模型的超分效果


