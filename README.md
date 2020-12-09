

# YES24_도서추천모델개발

## 0. 전처리

#### 0.1 파일 읽기 (JSON)

| 파일 이름    | 내용                             | 크기         |
| ------------ | -------------------------------- | ------------ |
| Click_Stream | 유저의 도서 클릭 정보 로그데이터 | (24105214,5) |
| Accounts     | 유저의 demography 정보           | (1741578,6)  |
| Products     | 도서의 상품 정보                 | (1745066,6)  |
| Orders       | 유저의 도서 구매 정보            | (8382514,5)  |

##### 사용한 코드

> 각 4가지 json 파일에 대해 해당 작업을 수행하여 csv 파일로 만들었음

```python
import json
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

path = '경로'

# files: 디렉토리 안의 파일 목록을 리스트로 만듭니다.
files = [f for f in listdir('디렉토리 경로') if isfile(join('디렉토리 경로', f))]

# json_files: 파일 목록을 파일 이름으로 하여 DataFrame을 만듭니다
json_files = pd.DataFrame({"file_id" : files})

# 초기 dataframe 칼럼을 선언해주기 위해 0번째 값을 초기화 시킵니다.
with open('디렉토리 경로'+json_files['file_id'][0], encoding = 'utf-8') as f:
        DF = pd.DataFrame(json.loads(line) for line in f)

# json file 목록을 읽으며 각 파일의 json 형태로 저장된 txt 파일을 DataFrame으로 형변환 한 후 concat를 통해 전체 데이터프레임 형성합니다.
for i in range(1,len(json_files)):
    with open('디렉토리 경로'+json_files['file_id'][i], encoding = 'utf-8') as f:
        tmp = pd.DataFrame(json.loads(line) for line in f)
    DF = pd.concat([DF, tmp])
```



#### 0.2 데이터 별 전처리

##### Accounts

| 칼럼           | 설명                          | 자료형  |
| -------------- | ----------------------------- | ------- |
| accounts_id    | 유저의 아이디 (6자리 숫자)    | int64   |
| gender         | 성별 (F/M)                    | object  |
| age            | 나이 (연속형)                 | float64 |
| address        | 주소 (시/도  -  군/구  -  동) | object  |
| zip_code       | 우편번호                      | object  |
| last_login_dts | json time                     | int64   |

* 결측치 및 이상치 처리

  > data가 매우 많기 때문에...삭제합니다

  * dropna로 일괄 삭제
  * age 중 7세 이하 삭제
  * 성별 중 F/M이 아닌 모든 값 삭제

* 접속 일자[last_login_dts]

  * date_time 형식으로 변환
  * `*1000000`을 통해 우리가 알 수 있는 형식으로 값을 변환

* 성별[gender]

  * Label Encoding
  * M = 1, F=0

* 나이[Age]

  * Categorical Encoding
    * 학령기 등을 기준으로 나이를 나눔
    * ~ 12세 / 12~19세 / 20~29세 / 30~40세 / 40~50세 / 51~64세 / 65세 이상
    * 각각 초등학생 / 중,고등학생 / 대학,취준생 / 사회초년생 / 자녀를 둔 부모 / 퇴직 전 / 퇴직 후

* 주소

  * 수도권 / 비수도권 더미 인코딩
    * 인천, 경기, 서울 = 수도권

##### Orders

| 칼럼       | 설명                                    | 자료형  |
| ---------- | --------------------------------------- | ------- |
| order_id   | 주문 시 생성되는 id 값                  | int64   |
| account_id | 주문한 유저의 id 값                     | int64   |
| product_id | 주문된 도서의 id 값                     | int64   |
| price      | 할인 등 기타 사항이 반영된 최종 결제 값 | float64 |
| created_at | java time                               | int64   |

* 결측치 삭제
* created_at
  * 주문 시간의 경우 java_time 변환
    * 초 = java_time / 1000
    * datetime.datetime.fromtimestamp(java_time/1000)
    * Y-m-d 형식으로 변환

##### Products

| 칼럼         | 설명           | 자료형    |
| ------------ | -------------- | --------- |
| product_id   | 도서의 id 값   | int64     |
| product_name | 도서 제목      | object    |
| category_id  | 도서 분류 id값 | int64     |
| published_at | 도서 출판 일   | date-time |
| shop_price   | 도서 정가      | float64   |
| maker_name   | 출판사 이름    | object    |

* 결측치 삭제
* 출판일
  * 이상치 확인하여 삭제
    * 8자리가 되지 않거나
    * 13월로 표기되거나
    * 0인 값
  * `from dateutil.parser import parse`
    * parse기능을 사용하여 하이픈(-) 형식으로 변환
  * Categorical Encoding
    * 3개월 미만 / 3개월 ~ 6개월 / 6개월 ~ 1년 / 1년 ~ 3년 / 3년 이후
* 가격
  * 배송 가능한 가격, 할인 가능한 가격, 전공 서적의 가격 등의 정보를 반영
    * 10,000원 미만 / 10,000원 ~ 14,000원 / 14,000원 ~ 19,000원 / 19,000원 이상
* 도서 카테고리
  * 일련 체계 상 뒷 3자리가 동일한 층위에 해당함
    * 인덱싱을 통해 처리 `[1:]`

##### Click

| 칼럼              | 설명                      | 자료형    |
| ----------------- | ------------------------- | --------- |
| request_date_time | 클릭 이벤트가 발생한 시간 | date-time |
| acount_id         | 유저 id                   | int64     |
| device_type       | 접속한 기기 정보 [M/P]    | object    |
| product_id        | 도서 id                   | int64     |
| before_product_id | 직전에 클릭한 도서 id     | int64     |

* Device type
  * 라벨 인코딩
  * Mobile:0, PC:1

#### 0.3 Feature Engineering

* orders & clicks 

  * 제공받은accounts, producdts 데이터가 click을 기준으로 하였기 때문에
  * orders 중 clicks에 없는 정보는 삭제하였음
  * YES24가 아닌 외부에서 구매된 상품인 경우에 해당함

* `New Preference`

  > 유저가 얼마나 신간 도서를 좋아하는가?

  * Click 데이터 기준, Product 데이터 결합
  * 2020년을 기준으로 신간 / 비신간 도서 구분
  * groupby method를 통해 account_id 별 신간 / 비신간 횟수 계산
  * Click 데이터의 account_id가 신간을 클릭한 횟수 / 비신간을 클릭한 횟수를 비율 계산
  * Account_id 별로 New Preference 선호 정도를 구할 수 있음

* `Category Preference`

  > 유저가 어떤 도서의 종류를 좋아하는가?

  * Click 데이터 기준, Product 데이터 결합
  * category 정보를 dummy coding 하여 칼럼 확장 [One-hot encoding]
  * Group by를 이용하여 account_id 별 category 클릭 횟수를 누적 (sum)
  * 그 중 가장 max인 cateogry 정보와 해당 클릭 횟수를 feature로 삼음

* `활동 시간`

  > 유저는 낮,밤 어느 시간 대를 선호하는가?

  * Click 데이터의 request_date_time을 이용
  * 한 개별 유저라도 낮 / 밤에 따라 선호하는 취향이 다를 것으로 가정함
  * 퇴근시간 7시를 기준으로 day / night을 나눔 [Onehot Encoding]
  * GROUPBY method를 이용하며 접속 시간을 day / night으로 더미 인코딩하여 account_id 별로 클릭 횟수를 누적(sum)
  * 비율을 계산하여 feature로 삼음

* `활동 요일`

  > 유저는 평일, 주말 중 어느 요일을 선호하는가?

  * Click 데이터의 request_date_time을 이용
  * 한 개별 유저라도 평일 / 주말에 따라 취향이 달라질 것으로 가정함
  * 월~목 = 평일 / 금~일 = 주말 로 Onehot Encoding
  * GROUPBY METHOD를 이용하여 Account_ID 별 평일/주말 클릭 횟수를 누적(sum)
  * 비율을 계산하여 각각 feature로 삼음

* `접속 기기`

  > 유저가 어떤 기기를 선호하는가?

  * Click 데이터의 device를 이용
  * 한 개별 유저라도 접속하는 기기에 따라 선호하는 취향이 달라질 것으로 가정
  * GROUPBY Method를 이용하여 접속 기기 별 클릭 횟수를 누적(sum)
  * 비율을 계산하여 각각 feature로 삼음

* `관여도`

  > 유저가 구매에 대해 얼마나 신경쓰는가?

  * Click 데이터와 Order 데이터의 차이를 이용
  * Click 횟수에 따라 Order 횟수를 나누어 비율 Feature를 생성
  * Order횟수를 Click횟수로 나누어 구매당 클릭 비율을 생성하여 Feature로 선정

* `BestSeller`

  > 도서의 베스트 셀러 유무가 유저의 구매에 영향을 미치는가?

  * 새로운 BestSeller 정보가 담긴 데이터를 요청
  * Click 기준으로 Best Seller 를 클릭한 횟수를 누적
  * 총 클릭 수에 대한 Best Seller 클릭 수를 비율 값으로 Feature를 선정

#### 0.4 전처리 결과물 [Dummy 데이터 - 향후 결합]

* User dummy
  * Accounts_id
  * Gender
    * Gender 0: 남성
    * Gender 1: 여성
  * Age
    * Age 0 : 12세 이하
    * ..
    * Age 6: 65세 이상
  * Category Preference
    * pref 1: 수험서 자격증
    * ...
    * pref 34: 에세이
  * New Preference
    * pref 0: 올해(2020)
    * pref 1: 과거(2020년 이전)
  * Address
    * Address 0:  수도권
    * Address 1: 비수도권
* Book dummy
  * Product_id
  * Category
    * cat 1: 수험서 자격증
    * ...
    * cat 34: 에세이
  * Published date
    * pub 0: 3개월 미만
    * ...
    * pub 4: 3년 이후
  * Price
    * price 0: 10,000원 미만
    * ...
    * Price 3: 19,000원 이상



## 1. Context Vector

#### 1.1 K-MEANS

* User, Book Feature 데이터를 K-means 기준으로 적절한 군집개수 정함
* Clustered data
  * User Cluster: 6
  * Book Cluster: 5
  * 필드 구성
    * User account - Book product - User Cluster - Book  Cluster - Purchased

#### 1.2 정규화

* User / Product 칼럼에 대한 정규화
  * K-means 기준으로 Aggregate
  * 정규화



* 발표 대본

  모델 설명에 앞서 모델의 인풋값으로 들어간 데이터의 형태에 대해서 잠시 설명드리자면, User와 Book Feature를 이와 같이 더미변수화하였고, 더미화로 인해 생기는 sparse matrix문제를 정규화를 통해 해결하였습니다.
  그리고 개별 책과 개별 사람을 고려하는 데는 너무나 방대한 연산과 cold-start문제가 발생하기 때문에, book feature와 user feature를 가지고 k-means를 통해 각각 book cluster와 user cluster를 이와 같이 만들었습니다.

## 2. 모델링

* 수도 코드

  * 5 그룹의 책 cluster 별로 context feature에 대한 prior distribution 설정
  * 새로운 context 발생 시 현재 distribution에서 coefficient sampling
  * Context 와 sampling 결과 값을 계산하여 가장 높은 scalar 값을 나타내는 책 Cluster 선택
  * 선택된 책 

* 발표 대본 설명

  먼저 책 cluster별로 각 context의 prior distribution을 가정한 후, 새로운 context가 발생 시 현재까지의 distribution에서 sampling 합니다. 그리고 context와 sampling 결과 값을 계산하여 가장 높은 scalar값을 나타내는 책 cluster를 선택합니다. 마지막으로 선택된 책 cluster의 reward값이 0인지, 1인지를 관측 후 결과에 따라 해당 책의 cluster의 distribution을 업데이트 하며 점차 개인맞춤형 책 cluster를 추천하는 방향으로 나아가는 모델입니다.



### Simple_Model

#### 유저 - 도서 그룹 베타 분포 매칭

* 베타 분포 with Thompson Sampling
  * 어떤 사건에 대해 성공과 실패에 의해 결정되는 분포
  * 톰슨 샘플링의 경우, 해당 분포를 사전 분포로 가정하여 특정한 사건의 관찰 및 결과에 따라 사후 분포를 누적함
* 적용 과정
  * User group [0~5]
  * Book group [0~4]
  * 각각의 페어에 대해 클릭 - 구매를 기준으로 성공과 실패를 나눔
  * 클릭 > 구매: Success
    클릭 > 비구매: Fail

|  book_cluster |    0.0 |    1.0 |    2.0 |    3.0 |    4.0 |
| ------------: | -----: | -----: | -----: | -----: | -----: |
| user0_success |  183.0 |  239.0 |  169.0 |  432.0 |  251.0 |
|    user0_fail |  435.0 |  590.0 |  488.0 | 1036.0 |  579.0 |
| user1_success |   15.0 |   13.0 |   25.0 |  551.0 |   42.0 |
|    user1_fail |   51.0 |   75.0 |   92.0 |  821.0 |  110.0 |
| user2_success |  154.0 |  262.0 |  160.0 |  388.0 |  240.0 |
|    user2_fail |  291.0 |  546.0 |  411.0 |  795.0 |  476.0 |
| user3_success |  641.0 | 1047.0 |  586.0 | 1467.0 |  771.0 |
|    user3_fail | 1265.0 | 2212.0 | 1618.0 | 3023.0 | 1831.0 |
| user4_success |  140.0 |  209.0 |  113.0 |  328.0 |  181.0 |
|    user4_fail |  259.0 |  397.0 |  294.0 |  618.0 |  361.0 |
| user5_success |  119.0 |  198.0 |  126.0 |  210.0 |  217.0 |
|    user5_fail |  227.0 |  412.0 |  287.0 |  522.0 |  401.0 |

* 상기 정보를 통한 업데이트 코드

```python
# 베타 분포를 학습

# 유저 천 명에 대한 베타 분포
# book_history 배열은 유저 클러스터에 대해 가장 큰 성공 확률을 주는 book cluster를 저장하는 배열
book_history = []

for i in range(len(user_1000)):
    # 유저 1000명 대상
    cluster = user_1000.iloc[i][1]
    
    beta_history = []
    
    for arm in range(0,5):
        # cluster에 대해 사후 베타 분포 확률 값을 샘플링
        # [앞서 정의한 유저 클러스터 - 북 클러스터 간의 성공/실패 누적 횟수를 이용]
        beta_sample = beta.rvs(user_beta.iloc[cluster*2][arm] + 1, user_beta.iloc[cluster*2+1][arm] + 1)
        beta_history.append(beta_sample)
    # 그 중 가장 큰 샘플링 값을 주는 book group을 인덱싱함
    book_cluster = np.argmax(beta_history)
    # book history에 해당 max idx의 count를 저장함
    book_history.append(book_cluster)
```

* 이에 대한 결과

|  Try | Beta | Real | Reward(Purhcase) |
| ---: | ---: | ---: | ---------------- |
|    0 |    1 |  4.0 | 0.0              |
|    1 |    0 |  2.0 | 0.0              |
|    2 |    0 |  2.0 | 0.0              |
|    3 |    0 |  3.0 | 0.0              |
|    4 |    0 |  3.0 | 0.0              |
|  ... |  ... |  ... | ...              |
| 4128 |    3 |  3.0 | 0.0              |
| 4129 |    3 |  3.0 | 1.0              |
| 4130 |    3 |  3.0 | 0.0              |
| 4131 |    3 |  3.0 | 0.0              |
| 4132 |    3 |  3.0 | 0.0              |

* 베타 분포의 기대값과 실제 구매 그룹의 비율 정보

| 그룹 | 베타 추천 비율 | 실제 구매 비율 |
| ---- | -------------- | -------------- |
| 0    | 5.44%          | 1.50%          |
| 1    | 7.83%          | 4.08%          |
| 2    | 0.04%          | 2.70%          |
| 3    | 3.17%          | 6.79%          |
| 4    | 5.87%          | 1.98%          |

#### Simple Linear posterior

* 단순화된 베이즈 선형 회귀
  * 사전 분포를 특별히 가정하지 않고 정규분포로 상정함
* 코드

```python
# narmas: 책 그룹의 개수, 유저 그룹를 input으로 받아 책 그룹을 추천했을 때 성공한 reward
# ndims: context vector의 차원 수, feature 수, 설명 변수의 수
narms = 5
ndims = 72
v = 0.9

# 규제 파라미터가 없는 공분산 행렬 Eye (I)
b = np.identity(ndims)
B = np.vstack((b,b,b,b,b))

# 평균 회귀 계수
mu_hat = np.zeros((ndims,narms))

# X.T * Y의 누적, 실제 reward와 예측 reward의 차이를 구하기 위한 매게변수
f = np.zeros(ndims)
F = np.vstack((f,f,f,f,f))

arm_number = reward['book_cluster'].iloc[:1200].astype('int')

bounds = np.zeros(narms)
```

```python
for i in context_1000.index:
    
    j = arm_number[i]
    
    # 사후 분포를 통해 계산된 회귀 계수 추정치 Mu_hat
    mean = np.transpose(mu_hat)[j]
    # 람다 * Eye의 Inverse, 업데이트된 사후 분포의 공분산
    covariance_matrix = ((v)**2)*np.linalg.inv(B[ndims*j:ndims*j+ndims])
    
    # 회귀 계수 sampling   in multivarate_normal 분포 (가우스 정규분포)
    samples = np.random.multivariate_normal(mean, covariance_matrix)
    sample_mu_tilde = np.expand_dims(samples,axis=1)
    
    arm_context = np.expand_dims(np.transpose(context_1000.iloc[i]),axis=1)
    # 누적된 추정치의 Reward값
    # X.T * 회귀 계수의 추정치 베타(B) = 추정치
    bounds[j] = bounds[j] + np.dot(np.transpose(arm_context),sample_mu_tilde)[0][0]
    
    # 업데이트
    # precision = [X.T * X + 람다 * Eye]
    # Cov 는 향후 Precision의 인버스 값
    B[ndims*j:ndims*j+ndims] = B[ndims*j:ndims*j+ndims] + np.dot(arm_context,np.transpose(arm_context))
    # F는 실제 리워드와 예측 리워드 bound의 차이를 계산하기 위한 업데이트 매게 변수
    # F = X.T * Y
    F[j] = F[j] + np.squeeze(arm_context)*reward_1000[i]
    # 추정 mu : (X.T* X + 람다* Eye).Inverse * (X.T * Y)
    # Y = X.T * B에서, B: (X.T * X + 람다 I).invese * (X.T * Y)에 해당하는 값
    np.transpose(mu_hat)[j] = np.dot(np.linalg.inv(B[ndims*j:ndims*j+ndims]),F[j])
```

* 기본 컨셉

  * Y = X.T * B + E

    * `X는 Context`로 고정 된 값
    * `Y는 Reward`
    * 추정 파라미터 B의 분포는 E의 분포를 따름

  * 평균 mu_hat 과 공분산 cov를 따르는 정규분포

    * 이때 사전 분포로 평균은 0, 분산/공분산은 Identity 행렬을 가정
    * 람다 규제 효과를 위해 v 파라미터 [설명 변수에 대한 규제] 설정
    * cov matrix = precision 형태로 삽입됨

  * 예측 치 bounds

    ```
    array([ 26.91224399,  55.85019365,  -3.1322682 , 106.50657819,
            69.36519875])
            
    array([ 0.02691224,  0.05585019, -0.00313227,  0.10650658,  0.0693652 ])
    ```

  * 업데이트

    * Precision B (분산조정)
      * X.T*X + (Identity/v**2).inverse
    * Mu_hat (평균 조정)
      * F = X.T * Y
      * B.inverse * F

* 결과

  | book_cluster | purchase | expect |
  | -----------: | -------: | ------ |
  |          0.0 |    0.036 | 0.026  |
  |          1.0 |    0.115 | 0.055  |
  |          2.0 |    0.023 | -0.003 |
  |          3.0 |    0.111 | 0.106  |
  |          4.0 |    0.049 | 0.069  |

### linear Full posterior Thompson Sampling

#### 유저 - 도서 그룹 베타 분포 매칭

* 베타 분포 with Thompson Sampling
  * 어떤 사건에 대해 성공과 실패에 의해 결정되는 분포
  * 톰슨 샘플링의 경우, 해당 분포를 사전 분포로 가정하여 특정한 사건의 관찰 및 결과에 따라 사후 분포를 누적함
* 적용 과정
  * User group [0~5]
  * Book group [0~4]
  * 각각의 페어에 대해 클릭 - 구매를 기준으로 성공과 실패를 나눔
  * 클릭 > 구매: Success
    클릭 > 비구매: Fail

- 선형 회귀 방식 이용
  $$
  Y=X^{T} \beta+\epsilon
  $$
  ​	

  ![Screen Shot 2020-12-09 at 10.53.32 AM](https://tva1.sinaimg.cn/large/0081Kckwgy1glhda1otzqj30vc09gq4d.jpg)

  - 기본 컨셉	

    - $X$는 `context`로 t 시점마다 새로운 context $X_{t} \in \mathbf{R}^{d}$가 모델에 입력됨
    - $Y$는 `reward`
    - $\beta$는 t시점의 $\mu$를 평균, $\sigma^2\Sigma_{t}$를 공분산으로 하는 정규분포를 따름
    - 이 때 $\sigma^2$은 t시점의 hyperparameter a를 평균, b를 분산으로 하는 Inverse gamma 분포를 따름

  - 업데이트

    - 평균

      - $\mu_{t}$는 $\sigma_{t}(\Lambda_{0}\mu_{0} + X^{T}Y)$로 업데이트
      - 이때 $\Lambda_{0}\mu_{0}$는 규제

    - 분산

      - $\Sigma_{t}$는 $(X^TX + \Lambda_{0})^-1$
      - $a_{t}$는 $a_{0} + t/2$로 업데이트, $b_{t}$는 $b_{0} + 1/2(Y^{T}Y+\mu^{T}_{0}\Sigma_{0}\mu_{0} -\mu_{t}^{T}\Sigma_{t}^{-1}\mu_{t})$

      ```python
  class LinearFullPosteriorSampling(BanditAlgorithm):
        """Thompson Sampling with independent linear models and unknown noise var."""
      
        def __init__(self, name, hparams):
          """Initialize posterior distributions and hyperparameters.
          Assume a linear model for each action i: reward = context^T beta_i + noise
          Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise
          level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,
          and precision matrices are initialized, and the ContextualDataset created.
          Args:
            name: Name of the algorithm.
            hparams: Hyper-parameters of the algorithm.
          """
      
          self.name = name
          self.hparams = hparams
      
          # Gaussian prior for each beta_i
          self._lambda_prior = self.hparams.lambda_prior
      
          self.mu = [
              np.zeros(self.hparams.context_dim + 1)
              for _ in range(self.hparams.num_actions)
          ]
      
          self.cov = [(1.0 / self.lambda_prior) * np.eye(self.hparams.context_dim + 1)
                      for _ in range(self.hparams.num_actions)]
      
          self.precision = [
              self.lambda_prior * np.eye(self.hparams.context_dim + 1)
              for _ in range(self.hparams.num_actions)
          ]
      
          # Inverse Gamma prior for each sigma2_i
          self._a0 = self.hparams.a0
          self._b0 = self.hparams.b0
      
          self.a = [self._a0 for _ in range(self.hparams.num_actions)]
          self.b = [self._b0 for _ in range(self.hparams.num_actions)]
      
          self.t = 0
          self.data_h = ContextualDataset(hparams.context_dim,
                                          hparams.num_actions,
                                          intercept=True)
      
        def action(self, context):
          """Samples beta's from posterior, and chooses best action accordingly.
          Args:
            context: Context for which the action need to be chosen.
          Returns:
            action: Selected action for the context.
          """
      
          # Round robin until each action has been selected "initial_pulls" times
          if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions
      
          # Sample sigma2, and beta conditional on sigma2
          sigma2_s = [
              self.b[i] * invgamma.rvs(self.a[i])
              for i in range(self.hparams.num_actions)
          ]
      
          try:
            beta_s = [
                np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
                for i in range(self.hparams.num_actions)
            ]
          except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling from {}.'.format(self.name))
            print('Details: {} | {}.'.format(e.message, e.args))
            d = self.hparams.context_dim + 1
            beta_s = [
                np.random.multivariate_normal(np.zeros((d)), np.eye(d))
                for i in range(self.hparams.num_actions)
            ]
      
          # Compute sampled expected values, intercept is last component of beta
          vals = [
              np.dot(beta_s[i][:-1], context.T) + beta_s[i][-1]
              for i in range(self.hparams.num_actions)
          ]
      
          return np.argmax(vals)
      
        def update(self, context, action, reward):
          """Updates action posterior using the linear Bayesian regression formula.
          Args:
            context: Last observed context.
            action: Last observed action.
            reward: Last observed reward.
          """
      
          self.t += 1
          self.data_h.add(context, action, reward)
      
          # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
          x, y = self.data_h.get_data(action)
      
          # The algorithm could be improved with sequential update formulas (cheaper)
          s = np.dot(x.T, x)
      
          # Some terms are removed as we assume prior mu_0 = 0.
          precision_a = s + self.lambda_prior * np.eye(self.hparams.context_dim + 1)
          cov_a = np.linalg.inv(precision_a)
          mu_a = np.dot(cov_a, np.dot(x.T, y))
      
          # Inverse Gamma posterior update
          a_post = self.a0 + x.shape[0] / 2.0
          b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
          b_post = self.b0 + b_upd
      
          # Store new posterior distributions
          self.mu[action] = mu_a
          self.cov[action] = cov_a
          self.precision[action] = precision_a
          self.a[action] = a_post
          self.b[action] = b_post
      
        @property
        def a0(self):
          return self._a0
      
        @property
        def b0(self):
          return self._b0
      
        @property
        def lambda_prior(self):
          return self._lambda_prior
      ```
      
      
      
      
      
      

- mu, cov 학습 및 현재 책의 pool의 추천값 계산

```python
#train data에서 학습한 mu, cov 불러오기
mu = np.load('~.npy')
cov = np.load('~.npy')

#hyper parameter 설정
num_actions = 5
hparams = tf.contrib.training.HParams(num_actions=num_actions)
hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,                                            context_dim=context_dim,
a0=6, #a
b0=6, #b
lambda_prior=0.25, #초기 람다값(규제)
initial_pulls=0,
mu=mu, #학습한 mu
cov=cov) #학습한 공분산
```

```python
#algorithm > linearFullPosteriorSampling 사용
algos = [LinearFullPosteriorSampling('LinFullPost', hparams_linear)]

t_init2 = time.time()
results2 = run_contextual_bandit(context_dim, num_actions, dataset2, algos)
```

- 추천값 계산한 것을 전처리 후 최종 추천리스트 뽑음

```python
#각 시점에 arm의 prior와 context가 내적된 값과 각 시점에서 선택된 arm 번호를 dataframe화
val_df = pd.DataFrame()
for i in range(len(new_vals)):
    val_df = pd.concat([val_df,pd.DataFrame(new_vals[i]).T],axis=0)

val_df['val_max'] = val_df[[0,1,2,3,4]].max(axis=1)
val_df_fin = val_df.reset_index().drop(columns='index')
val_df_fin.idxmax(axis=1).rename('idx')

val_df_fin = pd.concat([val_df_fin, val_df_fin.idxmax(axis=1).rename('val_max_idx')],axis=1)

#df_100_fin과 merge
df_fin_fin = pd.concat([df_100_fin[['account_id','product_id','book_cluster']],val_df_fin],axis=1)

df_fin_fin_fin = pd.DataFrame()
for i in range(len(df_fin_fin)):
    if df_fin_fin['book_cluster'][i] == df_fin_fin['val_max_idx'][i]:
        df_fin_fin_fin = pd.concat([df_fin_fin_fin,df_fin_fin.iloc[i,:]],axis=1)
df_fin_fin_fin = df_fin_fin_fin.T

#100명에 해당하는 account_list 
account_list = df_fin_fin_fin['account_id'].unique()

#1명당 24권의 리스트 추출
recc_list = []
for i in range(len(account_list)):
   recc_list.append([df_fin_fin_fin[df_fin_fin_fin['account_id']==account_list[i]].sort_values('val_max',ascending=False)['product_id'].tolist()[0:24]])
    
#최종 추천리스트 전처리 

#이중리스트 풀기
recc_book_list = [y for x in recc_list for y in x]
recc_book_list = [y for x in recc_list for y in x]
#중복된 값 지우기
recc_book_list = list(set(recc_book_list))
#기준이 된 책 지우기("내가 원하는 것을 나도 모를 때")
recc_book_list.remove(85927898)
```

- 전환치 도출

  ```python
  #추천된 책이 실제 구매되었는지 확인하여 1이면 sum >> 최종 전환치 도출
  df_100_fin[df_100_fin['product_id'].isin(recc_book_list)]['purchase'].sum()
  ```

  

## 3.평가

* off-policy

  * 임의의 책 1권 선정 ["내가 원하는 것을 나도 모를 때"]
    * 해당 책을 Click한 User 중 Random User 100 명 선정
    * 이에 따른 24권의 도서 선정

* T/S 적용 1번 방식

  첫번째 톰슨 샘플링 모델의 경우 user 100명을 선정하는 것까지는 YES24와 동일한 상황을 가정하고, 

  유저 클러스터에 대한 책 cluster의 구매 여부에 따른 베타 분포를 학습합니다. 

  그리고 click event가 발생 시 유저 cluster와 책 cluster의 확률 분포에서 임의의 값을 sampling합니다. 

  그리고 해당 user cluster에 대해 가장 큰 sampling 값을 갖는 책 cluster를 선택합니다. 

  그 후 선택된 책 cluster의 책들에 대해 학습한 context의 coeff를 내적하여 유저별 상위 24개를 추출합니다.

   마지막으로 실제 사용자가 구매한 책 중 이 모델이 추천한 책이 포함된 값을 계산합니다.

* T/S 적용 2번 방식

  두번째 톰슨 샘플링 모델의 경우 아까와 같이 user 100명 선정하는 단계까지는 동일하며, 

  첫번째 모델처럼 train data에서 책 cluster별 context의 coeff를 학습하는데, 

  다만 선택된 책의 pool 내에서만 뽑는 것이 아니라 train data에 있는 전체 책에 대해 학습합니다. 

  test data에서는 click event 발생시 학습된 feature별 확률 분포에서 임의 값을 sampling하고, 

  sampling한 값을 click event마다 생기는 context feature와 내적하여 가장 큰 값을 갖는 유저 당 상위 24권을 추출합니다.





# 참조링크

---

```markdown
https://darkpgmr.tistory.com/119
http://norman3.github.io/prml/docs/chapter03/3
https://ratsgo.github.io/statistics/2017/05/31/gibbs/
https://4four.us/article/2014/11/markov-chain-monte-carlo
https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
https://www.secmem.org/blog/2019/01/11/mcmc/
https://excelsior-cjh.tistory.com/193
```



