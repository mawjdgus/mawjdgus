출처 : https://deep-learning-study.tistory.com/529

Using Xception to classificate cat and dog


**Xception**은 Inception 모듈에 대한 고찰로 탄생한 모델입니다. 

Xception은 완벽히 **cross-channel correlations와 spatial correlations를** ***독립적*** **으로 계산**하기 위해 고안된 모델입니다. 

아래 그림은 Inception v3에서 사용하는 일반적인 Inception 모듈 입니다.

![image](https://user-images.githubusercontent.com/67318280/132187784-0049f0d5-5339-41e2-9fbb-4224903a22d6.png)

이를 간소화 하면 아래와 같습니다.

![image](https://user-images.githubusercontent.com/67318280/132187936-1de73982-3d2e-4ac4-a1c3-f6d59a91ff4b.png)

1x1 conv 이후에 3x3 conv 연산이 수행되는 구조입니다.

- cross-channel correlations
- spatial correlations

둘을 독립적으로 수행한다고 합니다. 

- 1x1 conv는 cross-channel correlation을 계산
- 3x3은 spatial correlations를 계산

Inception 모듈이 좋은 성능을 나타내는 이유는 cross-channel correlations와 spatial correlations를 잘 분해해서 계산했기 때문이라고 합니다. 

Xception은 완벽히 cross-channel correlations와 spatial correlations를 독립적으로 계산하고 mapping하기 위해 고안된 모델입니다.

**2. Depthwise Separable Convoltuion**

다음에는 Inception 모듈과 작동방식이 비슷한 

**Depthwise Separable Convolution**

Depthwise Separable Convolution은 **Depthwise Convolution 이후에 Pointwise Convolution을 수행**합니다. 두 가지 그림을 가져왔습니다.

![image](https://user-images.githubusercontent.com/67318280/132188345-f94f4537-b7d0-4d00-8595-5dd7eb2675cd.png)

![image](https://user-images.githubusercontent.com/67318280/132188360-efa01eca-09df-42f1-94f5-9d0c154e4239.png)

Depthwise Convolution은 입력 채널 각각에 독립적으로 3x3 conv를 수행합니다. 

입력 채널이 5개이면 5개의 3x3 conv가 연산을 수행하여, 각각 입력값과 동일한 크기 피쳐맵을 생성합니다. 

그리고 각 피쳐맵을 연결하여 5개 채널의 피쳐맵을 생성합니다. 

Pointwise Convolution은 모든 채널에 1x1 conv를 수행하여, 채널 수를 조절하는 역할을 합니다. 

이렇게 연산을 수행하면 연산량이 감소합니다.

 
**3. Modified Depthwise Separable Convolution(Extreme Inception)**

Xception은 Depthwise Separable Convolution을 수정해서 inception 모듈 대신에 사용합니다. 

그리고 Extreme Inception이라고 부릅니다.

아래 구조를 활용하면 Inception 모듈보다 효과적으로 cross-channels correlations와 spatial correlations를 독립적으로 계산할 수 있습니다.

![image](https://user-images.githubusercontent.com/67318280/132188995-73db4239-5a5b-42fb-be3c-23127e70cd4d.png)

![image](https://user-images.githubusercontent.com/67318280/132189003-39f9d4da-6b82-4253-8d5f-cef8fa725117.png)


pointwise convolution 이후에 depthwise convolution을 사용합니다.

입력값에 1x1 conv를 수행하여 채널 수를 조절합니다. 

그리고 채널 수는 n개의 segment로 나눠집니다. 

이 n은 하이퍼파라미터 입니다. 예를 들어, 100개의 채널 수가 3~4개의 segment로 나눠집니다. 

나눠진 segment 별로 depthwise convolution(3x3 conv)를 수행합니다. 

각 출력값은 **concatenate** 됩니다.


**(1) 연산의 순서가 다릅니다.**

기존 depthwise separable convolution은 depthwise convolution(3x3 conv)를 먼저 수행하고 pointwise convoltion(1x1 conv)를 수행합니다. 

수정된 버전은 pointwise convoltuion(1x1 conv)를 수행하고, depthwise convolution(3x3 conv)를 수행합니다. 

 

**(2) 비선형 함수의 존재 유무입니다.**

Inception 모듈은 1x1 conv 이후에 ReLU 비선형 함수를 수행합니다. 

하지만 Xception에서 사용하는 모듈은 비선형 함수를 사용하지 않습니다. 아래는 실험 결과입니다. 

![image](https://user-images.githubusercontent.com/67318280/132189155-d670ae9c-5f11-4f85-af42-eeb8b3c970ab.png)

**Xception architecture**

Xception은 14개 모듈로 이루어져있고, 

**총 36개의 convolutional layer가 존재**합니다. 

그리고 residual connection을 사용합니다. 

아래 그림을 보면 구조를 세부적으로 쪼개서 나타냅니다. 

입력값은 Entry flow 거쳐서 **midlle flow를 8번 거칩니다.** 

그리고 exit flow를 통과합니다.

![image](https://user-images.githubusercontent.com/67318280/132189245-03b32560-29ce-4826-93b3-85ba21c05f8f.png)

그리고 이놈을 이용해 Cat Dog 분류기를 만들어 봤습니다.








