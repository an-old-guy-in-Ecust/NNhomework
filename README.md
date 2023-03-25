直接运行hw1.ipynb即可！

（1）定义双层神经网络模型
前向计算：

1. 输入X，$X\in R^{1\times 784}$
2. 第一层输出：$Y_1=XW_1+b_1,o_1=sigmoid(Y_1)$
3. 第二层输出：$Y_2=o_1 W_2+b_2,o_2=softmax(Y_2)$
   误差损失函数：交叉熵加正则化项
   $L = -\sum_t y_t\log o_{2t}+\lambda(||W_2||_2+||b_2||_2)$

后向传播：

1. $-\frac{dL}{do_2}=-\frac{1}{o_2}$
2. $\frac{dL}{dY_2}=-\frac{1}{o_2}o_2(1-o_2)=o_2-1$
3. $\frac{dL}{do_1} = \frac{dL}{dY_2}\frac{dY_2}{do_1}=(o_2-1)W_2^T$
4. $\frac{dL}{dW_2} = \frac{dL}{dY_2}\frac{dY_2}{dW_2}=(o_2-1)o_1^T$
5. $\frac{dL}{dY_1}=(o_2-1)W_2^T o_1(1-o_1)$
6. $\frac{dL}{dW_1} = \frac{dL}{dY_1}\frac{dY_1}{dW_1}=(o_2-1)W_2^T o_1(1-o_1)X$

   梯度更新：对于参数$W_1,W_2,b_1和b_2$，设置学习率进行梯度下降，学习率设置为每1000个batch衰减为95%。
   batch_size设置为32。
