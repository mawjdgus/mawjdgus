.
출처 : https://gaussian37.github.io/dl-concept-batchnorm/

## BatchNormalization이란 뭘까..

## 목차
- Batch
- Internal Covariant Shift
- Batch Normalization
- Internal Covariant Shift (more)
- Batch Normalization (effection)
- Training Phase, Testing Phase Batch Nrom
- Fully Connected Layer nad Batch Normalization
- Convolution Layer and Batch Normalization
- Limited of Batch Normalization
- using in pytorch

## Batch

배치 정규화를 설명하기 전에, batch에 대해 설명하자면,


- 학습 데이터 전부를 넣어서 gradient를 다 구하고 그 모든 gradient를 평균해서 한번에 모델을 업데이트합니다.

- 이런 방식으로 하면 대용량의 데이터를 한번에 처리하지 못하기 때문에 데이터를 batch 단위로 나눠서 학습을 하는 방법을 사용하는 것이 일반적입니다.

- 그래서 사용하는 것이 stochastic gradient descent 방법입니다.
- SGD에서는 gradient를 한번 업데이트 하기 위하여 일부의 데이터만을 사용합니다.
즉 batch size만큼만 사용합니다.
- 위 gradient descent의 식을 보면 시그마 안의 분모가 B이며 이는 batch의 크기가 됩니다.

## Internal Covariant Shift


- Batch 단위로 학습을 하게 되면 발생하는 문제점이 있는데 이것이 논문에서 다룬 Internal Covariant Shift 입니다.
- 먼저 Internal Covariant Shift의 의미를 알아보면 학습 과정에서 계층 별로 입력의 데이터 분포가 달라지는 현상을 말합니다.
- 각 계층에서 입력으로 featrue를 받게 되고 그 feature는 convolution이나 fully connected연산을 거친 뒤 activation function을 적용하게 됩니다.
- 그러면 연산 전/후에 데이터 간 분포가 달라질 수 있습니다.
- 이와 유사하게 Batch 단위로 학습을 하게 되면 Batch 단위간에 데이터 분포의 차이가 발생할 수 있습니다.
- 즉, Batch간의 데이터가 상이하다고 말할 수 잇는데 위에서 말한 Internal Covariant Shift 문제입니다.
- 이 문제를 개선하기 위한 개념이 Batch Normalization입니다.

## Internal Covariant Shift(more)

- Neural Network에서는 많은 파라미터들에 의해서 학습에 어려움이 있습니다.
- 딥러닝에서는 학습의 어려움의 정도가 더 커졌는데 Hidden Layer의 수가 점점 더 증가하기 때문입니다.
- 딥러닝에서 Layer가 많아질 때 학습이 어려워지는 이유는 weight의 미세한 변화들이 가중이 되어 쌓이면 Hidden Layer의 깊이가 깊어질수록 그 변화가 누적되어 커지기 때문입니다.
- 예를들어 학습 중에 weight들이 기존과는 다른 형태로 변형될 수 있습니다.
- 즉 기존의 Hidden Layer와는 또 다른 Layer의 결과를 가지게 된다는 말입니다.
- 이를 Internal Covariant Shift라고 말합니다.
- 어떤 문제든지 Variance는 문제를 일으키곤 합니다.
- 각 Layer의 Input feature가 조금씩 변해서 Hidden Layer에서의 Input feature의 변동량이 누적되게 되면 각 Layer에서는 입력되는 값이 전혀 다른 유형의 데이터라고 받아들일 수 있습니다.
- 예를 들어 Training 셋의 분포와 Test 셋의 분포가 다르면 학습이 안되는 것과 같이 같은 학습과정 속에서도 각 Layer에 전달되는 feature의 분포가 다르게 되면 학습하는 데 어려움이 있습니다.
- 이 문제는 training 할 때 마다 달리젝 되고 Hidden Layer의 깊이가 깊어질수록 변화가 누적되어 feature의 분포 변화가 더 심해지게 됩니다.
-  weight에 따른 가중치가 되는 부분.
- Batch Normalization에서는 이러한 weight에 따른 가중치의 변화를 줄이는 것을 목표로 합니다.
- 따라서 Activation하기 전 값의 변화를 줄이는 것이 BN의 목표입니다.
