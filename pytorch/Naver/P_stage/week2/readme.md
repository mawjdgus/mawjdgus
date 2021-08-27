[P_stage 2주차]

ResNet 에 대해 알아보자

ResNet은 마이크로소프트에서 개발한 알고리즘이다.
ResNet의 original 논문명은 "Deep Residual Learning for Image Recognition"이고, 저자는 Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jiao Sun이다.

층수에 있어서 ResNet은 매우 깊은데
2014년의 GoogLeNet이 22개 층인데, ResNet은 152개 층을 갖는다.
![image](https://user-images.githubusercontent.com/67318280/131111608-522839ab-fcf5-49e6-a800-dac6fefd9915.png)

"망을 깊게하면 무조건 성능이 좋아질까?"
를 확인하기 위해 컨볼루션 층과 fully-connected 층으로 20층, 56층의 네트워크를 각각 만든 다음에 성능을 테스트해보았다. 

더 깊은 구조의 56층의 네트워크가 20층의 네트워크보다 더 나쁜 성능을 보임을 알 수 있다. 
기존의 방식은 망을 무조건 깊게 하면 좋아진다는 결론을 냈는데, 그게 아닌 것이다.

**Residual Block**

![image](https://user-images.githubusercontent.com/67318280/131111947-2378224f-85ad-4fd7-8ca8-0f09a13a4380.png)

ResNet은 F(x) + x를 최소화

F(x)를 0에 가깝게 만드는 것이 목적

F(x) = H(x) - x

잔차를 최소

![image](https://user-images.githubusercontent.com/67318280/131112153-e64d6197-4783-40ef-b488-e8eb25ee0dd0.png)

![image](https://user-images.githubusercontent.com/67318280/131112177-5af1c233-240d-47bb-ba84-d81ece29fe50.png)

