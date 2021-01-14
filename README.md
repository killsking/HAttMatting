@article{meng2020gia,
  title={GIA-Net: Global Information Aware Network for Low-light Imaging},
  author={Meng, Zibo and Xu, Runsheng and Ho, Chiu Man},
  journal={arXiv e-prints},
  pages={arXiv--2009},
  year={2020}
}

@article{kwon2020dale,
  title={DALE: Dark Region-Aware Low-light Image Enhancement},
  author={Kwon, Dokyeong and Kim, Guisik and Kwon, Junseok},
  journal={arXiv preprint arXiv:2008.12493},
  year={2020}
}

@inproceedings{yang2020fidelity,
  title={From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement},
  author={Yang, Wenhan and Wang, Shiqi and Fang, Yuming and Wang, Yue and Liu, Jiaying},
  booktitle=CVPR,
  pages={3063--3072},
  year={2020}
}

@inproceedings{atoum2020color,
  title={Color-wise Attention Network for Low-light Image Enhancement},
  author={Atoum, Yousef and Ye, Mao and Ren, Liu and Tai, Ying and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={506--507},
  year={2020}
}

@article{lv2019attention,
  title={Attention guided low-light image enhancement with a large scale low-light simulation dataset},
  author={Lv, Feifan and Li, Yu and Lu, Feng},
  journal={arXiv: 1908.00682},
  year={2019}
}
@inproceedings{guo2020zero,
  title={Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1780--1789},
  year={2020}
}

@inproceedings{wei2020physics,
  title={A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising},
  author={Wei, Kaixuan and Fu, Ying and Yang, Jiaolong and Huang, Hua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2758--2767},
  year={2020}
}

@article{fu2020learning,
  title={Learning an Adaptive Model for Extreme Low-light Raw Image Processing},
  author={Fu, Qingxu and Di, Xiaoguang and Zhang, Yu},
  journal={arXiv preprint arXiv:2004.10447},
  year={2020}
}
@article{wang2020extreme,
  title={Extreme Low-Light Imaging with Multi-granulation Cooperative Networks},
  author={Wang, Keqi and Gao, Peng and Hoi, Steven and Guo, Qian and Qian, Yuhua},
  journal={arXiv preprint arXiv:2005.08001},
  year={2020}
}

@article{karadeniz2020burst,
  title={Burst Denoising of Dark Images},
  author={Karadeniz, Ahmet Serdar and Erdem, Erkut and Erdem, Aykut},
  journal={arXiv preprint arXiv:2003.07823},
  year={2020}
}

@inproceedings{chen2018learning,
  title={Learning to see in the dark},
  author={Chen, Chen and Chen, Qifeng and Xu, Jia and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3291--3300},
  year={2018}
}

@article{xiong2020unsupervised,
  title={Unsupervised Real-world Low-light Image Enhancement with Decoupled Networks},
  author={Xiong, Wei and Liu, Ding and Shen, Xiaohui and Fang, Chen and Luo, Jiebo},
  journal={arXiv preprint arXiv:2005.02818},
  year={2020}
}
@article{liang2020deep,
  title={Deep Bilateral Retinex for Low-Light Image Enhancement},
  author={Liang, Jinxiu and Xu, Yong and Quan, Yuhui and Wang, Jingwen and Ling, Haibin and Ji, Hui},
  journal={arXiv preprint arXiv:2007.02018},
  year={2020}
}
@inproceedings{zhang2020attention,
  title={Attention-based network for low-light image enhancement},
  author={Zhang, Cheng and Yan, Qingsen and Zhu, Yu and Li, Xianjun and Sun, Jinqiu and Zhang, Yanning},
  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

@article{li2020visual,
  title={Visual Perception Model for Rapid and Adaptive Low-light Image Enhancement},
  author={Li, Xiaoxiao and Guo, Xiaopeng and Mei, Liye and Shang, Mingyu and Gao, Jie and Shu, Maojing and Wang, Xiang},
  journal={arXiv preprint arXiv:2005.07343},
  year={2020}
}

@article{zhang2020self,
  title={Self-supervised Image Enhancement Network: Training with Low Light Images Only},
  author={Zhang, Yu and Di, Xiaoguang and Zhang, Bin and Wang, Chunhui},
  journal={arXiv},
  pages={arXiv--2002},
  year={2020}
}
@inproceedings{xu2020learning,
  title={Learning to Restore Low-Light Images via Decomposition-and-Enhancement},
  author={Xu, Ke and Yang, Xin and Yin, Baocai and Lau, Rynson WH},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2281--2290},
  year={2020}
}
We study the reasons for why low resolution transmission map achieves the highest dehazing performance in deep. We obtain three reasons for this. First, the proposed model decompose the dehazing problem into two phases, which restores a coarse dehazed result and then refines the image details. With the help of low resolution transmission map, the proposed model can restore a high quality coarse dehazed result, and then the decoder can refine the image details with multi-scale GAN. Second, the high resolution transmission map contains much details than low resolution transmission map. The details are hard to be predicted accurately. If the transmission map is not correct, which will mislead the dehazing processing. As shown in Figure, due to the incorrect transmission map, the dehazed result contains haze. Third, predicting a high resolution transmission map accurately needs higher model capacity, which results in a more complexity model. The model proposed in this paper is also consider the execute time and design a fast dehazing model, which is not suitable for predicting a high resolution transmission map accurately. We also notice that the predicted high resolution transmission map contains some details, which is not presented in ground truth.

It should be pointed out that the experiments are conducted with similar architecture except the position of density generator. The architectures used in experiments have same depth of the pyramid, so the influence of the depth of the pyramid is eliminated. We also admit that the high depth of the pyramid boosts the dehazing performance. First, the high resolution transmission map contains much details than low resolution transmission map. The details are hard to be predicted accurately. If the transmission map is not correct, which will mislead the dehazing processing. As shown in Figure, due to the incorrect transmission map, the dehazed result contains haze. Second, predicting a high resolution transmission map accurately needs higher model capacity, which results in a more complexity model.

The start point of the proposed model is similar to ‘Karras et al.’, and we construct an image pyramid from coarse to fine scale. First, we obtain a coarse dehazed result with the help of haze density map. Second, we refine the dehazed result by adding the details. The different is that we use the coarse density map to restore the coarse dehazed result. At each level we add GAN to improve the dehazing quality similar to ‘Karras et al.’ We also test the influence of the density map generator. Without the help of density map generator, the coarse dehazed result contains some haze, which is hard to be removed by decoder and GAN. As we can see, with the help of density map generator, the coarse dehazed result is much cleaner, and with the help of GAN the image details can be recover processly.





We insert the proposed density generator to the decoder at different scales to generate transmission maps with different resolution. We show two examples of the model architecture in Figure. The only different is the position of the density generator in the decoder level. The experiments can show the influence of the different sizes transmission maps. The PSNR is used to show the differences of the generated dehazed result and the ground truth. As shown in Table, the dehazed results with low-resolution transmission map have higher PSNR valuses.




The reasons for low resolution transmission map achieves the highest dehazing performance is three aspects. First, the high resolution transmission map contains much details than low resolution transmission map. The details are hard to be predicted accurately. If the transmission map is not correct, which will mislead the dehazing processing. As shown in Figure, due to the incorrect transmission map, the dehazed result contains haze. Second, predicting a high resolution transmission map accurately needs higher model capacity, which results in a more complexity model. The model proposed in this paper is also consider the execute time and design a fast dehazing model, which is not suitable for predicting a high resolution transmission map accurately. We also notice that the predicted high resolution transmission map contains some details, which is not presented in ground truth. Third, the proposed model decompose the dehazing problem into two phases, which restores a coarse dehazed result and then refines the image details. This process reduces the complexity of dehazing problem.  





@article{li2019pdr,
  title={PDR-Net: Perception-inspired single image dehazing network with refinement},
  author={Li, Chongyi and Guo, Chunle and Guo, Jichang and Han, Ping and Fu, Huazhu and Cong, Runmin},
  journal={IEEE Transactions on Multimedia},
  volume={22},
  number={3},
  pages={704--716},
  year={2019},
  publisher={IEEE}
}
@article{li2018cascaded,
  title={A cascaded convolutional neural network for single image dehazing},
  author={Li, Chongyi and Guo, Jichang and Porikli, Fatih and Fu, Huazhu and Pang, Yanwei},
  journal={IEEE Access},
  volume={6},
  pages={24877--24887},
  year={2018},
  publisher={IEEE}
}



https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip
https://github.com/eezkni/UEGAN
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

SRResNet【Ledig】 和 EDSR【Lim】 通过残差模块提高超分模型的效果。SRDenseNet 则通过应用密集链接模块提高超分模型的效果。Zhang等人则通过结合全局残差和密集链接来提高深度模型的超分效果。DDBPN【Haris】 则通过反馈机制提高超分精度。Li等人则应用反馈机制提高模型的特征表达能力。为了得到不同尺度的特征表达，多个分支的网络被提出来并应用于超分模型的设计。Niu等人则通过通道和空间注意力机制提高模型的超分效果。

。
H. Ibrahim and N. S. P. Kong, “Brightness preserving dynamic histogram equalization for image contrast enhancement,” IEEE Transactions on Consumer Electronics, vol. 53, no. 4, pp. 1752–1758, 2007.
C. Lee, C. Lee, and C.-S. Kim, “Contrast enhancement based on layered difference representation of 2d histograms,” IEEE transactions on image processing (TIP), vol. 22, no. 12, pp. 5372–5384, 2013.


D. J. Jobson, Z.-u. Rahman, and G. A. Woodell, “A multiscale retinex for bridging the gap between color images and the human observation of scenes,” IEEE Transactions on Image processing, vol. 6, no. 7, pp. 965–976, 1997.

D. J. Jobson, Z.-u. Rahman, and G. A. Woodell, “Properties and performance of a center/surround retinex,” IEEE Transactions on Image processing, vol. 6, no. 3, pp. 451–462, 1997.
Z. Ying, G. Li, Y. Ren, R. Wang, and W. Wang, “A new low-light image enhancement algorithm using camera response model,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 3015–3022.

K. G. Lore, A. Akintayo, and S. Sarkar, “L lnet: A deep autoencoder approach to natural low-light image enhancement,” Pattern Recognition, vol. 61, pp. 650–662, 2017.
Chen Wei, Wenjing Wang, “Deep retinex decomposition for low-light enhancement,” British Machine Vision Conference (BMVC), 2018.

A.	Ignatov, N. Kobyshev, R. Timofte, K. Vanhoey, and L. Van Gool, “Dslr-quality photos on mobile devices with deep convolutional networks,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 3277–3285.
Z. Hui, X. Wang, L. Deng, and X. Gao, “Perception-preserving convolutional networks for image enhancement on smartphones,” in ECCV Workshop, 2018, pp. 197–213.

A.	Ignatov, N. Kobyshev, R. Timofte, K. Vanhoey, and L. Van Gool, “Wespe: weakly supervised photo enhancer for digital cameras,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2018, pp. 691–700.
Chen Chen, Qifeng Chen and V. Koltun, “Learning to see in the dark,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Ren, W., Liu, S., Ma, L., Qianqian, X., Xiangyu, X., Cao, X., Junping, D., Yang, M.-H.: Low-light image enhancement via a deep hybrid network,” IEEE Trans. Image Process. 28(9), 4364–4375 (2019)

Xu, K., Yang, X., Yin, B., & Lau, R. W., “Learning to Restore Low-Light Images via Decomposition-and-Enhancement,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
@inproceedings{wei2020physics,
   title={A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising},
   author={Wei, Kaixuan and Fu, Ying and Yang, Jiaolong and Huang, Hua},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2020},
 }
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}


这些基于直方图均衡化的暗光增强的区别主要在于应用不同的先验和限制。如BPDHE应用增强结果保留图像的亮度的动态得到增强结果，LDR则关注于每个层的直方图之间的表示，通过增大相邻像素之间的差异达到增强的目的。
有一些暗光增强的算法则是基于Retinex理论，这个理论假设一幅图像可以表示为反射层（R）和光照层（L）。MSR [11] and SSR [12]尝试通过恢复和使用光照层来到大暗光增强的目的. Ying等人通过双曝光策略和相机反应模型做暗光增强。
最近，基于深度学习的暗光增强算法成为学界主要的研究对象。很多工具如端到端模型和GAN模型已经被用于图像增强算法。LLNet使用多层感知机构造的自编码网络进行图像增强和去噪。RetinexNet则通过将Retinex理论和CNN结合去估计光照层并通过调节光照层达到暗光增强的目的。DPED通过合成感知损失函数（composite perceptual error function）设计了一个从低质手机照片转换为高质量照片的端到端增强算法。PPCN应用老师-学生策略学习一个紧凑的网络以减少推理时间。WESPE则基于弱监督提出了一个可以缓减成对数据集要求的暗光增强算法。Chen等人设计了一个从原始感光照片进行基于CNN的暗光增强算法。Ren等人提出了一个基于混合网络的暗光增强算法。Xu等人设计了一个将图像在频率域进行分解和增强的模型，从而避免在增强的过程中导致噪声放大的问题。Wei等人提出了一个从感光器件原始数据进行暗光增强的算法。Guo等人提出了一个无参考曲线估计暗光增强算法。


Deng, Q., Huang, Z., Tsai, C. C., & Lin, C. W. (2020, August). HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing. In European Conference on Computer Vision (pp. 722-738). Springer, Cham.
Dong et al, Multi-Scale Boosted Dehazing Network with Dense Feature Fusion. (CVPR)
Qin et al, FFA-Net: Feature Fusion Attention Network for Single Image Dehazing
Dong et al, FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing.
Shao et al, Domain Adaptation for Image Dehazing
Deng等提出了困难感知的去雾算法。Qin等人提出了一个特征注意力机制来提高去雾算法的性能。Dong提出了一个双判别器对抗生成网络去雾算法。Shao等人提出了一个域适应的去雾算法来提高去雾算法在自然图像上面的性能。Dong提出了一个基于boost和特征融合的去雾的去雾算法。

