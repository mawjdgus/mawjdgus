REFERENCE : https://bkshin.tistory.com/entry/dd?category=1042793

# NAIVE BAYES

## Bayesian Estimation

## 베이즈 추정이란 ?

협력업체로부터 납품받은 기계의 성능을 평가한다고 해보자. 납푸받은 몇 개의 부품을 랜덤 샘플링해 표본을 통해 얻은 정보로만 모수를 평가해야 한다. 하지만 과거 납품 시 성능검사기록이나 비슷한 부품의 성능자료, 이 부품의 물리적 특성에 관한 지식 등을 통해 이 부품의 사전 정보를 얻을 수도 있다. 이런 경우 단순히 표본을 통해 모수를 추정하기보다는 **표본 정보와 사전 정보를 함께 사용하여** 모수를 추정하는 것이 보다 바람직할 것이다.

이와 같이 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 사후 확률을 추론하는 통계적 방법을 베이즈 추정(Bayesian Estimation)이라고 합니다.

베이즈 추정은 아래와 같이 계산할 수 있다.

P(A|B) = P(B|A)P(A)/P(B)

사례를 들어보자. 어떤 마을 전체 사람들의 10.5%가 암 환자이고, 89.5%가 암 환자가 아니다. 이 마을의 모든 사람에 대한 암 검진을 실시했다고 하자. 암 검진시 양성 판정, 음성 판정 결과가 나올 수 있습니다. 하지만 검진이 100% 정확하지는 않고 약간의 오차가 있습니다. 암 환자 중 양성 판정을 받은 비율은 90.5%, 암 환자 중 음성 판정을 받은 비율은 9.5, 암 환자가 아닌 사람 중 양성 판정을 받은 비율은 20.4$, 암 환자가 아닌 사람 중 음성 판정을 받은 비율은 79.6%입니다. 어떤 사람이 양성 판정을 받았을 때 이 사람이 암 환자일 확률은 얼마일까?

약자 - C: Cancer(암 환자), P: Positive(양성), N: Negative(음성)
P(C): 암 환자일 확률 = 0.105
P(~C): 암 환자가 아닐 확률 = 0.895
P(P|C): 암 환자일 때 양성 판정을 받을 확률 = 0.905 (이를 민감도라고 합니다. sensitivity)
P(N|C): 암 환자일 때 음성 판정을 받을 확률 = 0.095
P(P|~C): 암 환자가 아닐 때 양성 판정을 받을 확률 = 0.204
P(N|~C): 암 환자가 아닐 때 음성 판정을 받을 확률 = 0.796 (이를 특이도라고 합니다. specificity)

이때 P(C|P): 어떤 사람이 양성 판정을 받았을 때 이 사람이 암 환자일 확률은?

베이즈의 추정에 의해

P(C|P) = P(P|C)*P(C) / P(P)

여기서, P(P) = P(P, C) + P(P, ~C) = P(P|C)*P(C) + P(P|~C)*P(~C)입니다.

P(P), 즉 양성 판정을 받을 확률은 암 환자이자 양성 판정을 받을 확률(P(P, C))과 암 환자가 아닌데 양성 판정을 받을 확률(P(P, ~C))의 합과 같습니다.

조건부 확률에 의해 P(P, C) = P(P|C)*P(C)이고, P(P, ~C) = P(P|~C)*P(~C)입니다.

따라서, P(C|P) = P(P|C)*P(C) / P(P) = 0.905*0.105 / (0.905*0.105 + 0.204*0.895) = 0.342입니다.

P(C|P), P(C|N), P(C|N)도 마찬가지로 계산하면 각각 0.65797, 0.013808, 0.986192가 나옵니다. 

암 환자는 양성 판정을 받을 확률이 높습니다. 하지만 양성 판정을 받았어도 암이 아닐 확률이 더 높다는 걸 알 수 있습니다.
