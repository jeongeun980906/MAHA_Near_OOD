# MAHA_Near_OOD

- Dataset
    - train with even MNIST
    - Near OOD: Odd MNIST
    - Far OOD: Letter MNIST (EMNIST)

### Maha distance
Paper URL: [MD](https://arxiv.org/abs/1807.03888.pdf)

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/MAHA_Near_OOD/blob/main/image/MD.png">
</p>

|      Data      |  AUROC |
| ---------------|--------|
|  In - NEAR OOD | 0.8473 |
|  In - FAR OOD  | 0.7912 |
| NEAR - FAR OOD | 0.6140 |

### Relative Maha distance
Paper URL: [RMD](https://arxiv.org/pdf/2106.09022.pdf)

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/MAHA_Near_OOD/blob/main/image/RMD.png">
</p>

|      Data      |  AUROC |
| ---------------|--------|
|  In - NEAR OOD | 0.7277 |
|  In - FAR OOD  | 0.8397 |
| NEAR - FAR OOD | 0.6650 |