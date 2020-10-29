# Model: Contextual Bandit with Thompson Sampling

> **Reference**
>
> Deep Bayesian Bandits showdown
>
> : An EMPRIRICAL COMPARISON OF BAYESIAN DEEP NETWORKS FOR THOMPSON SAMPLING
>
> > Carlos Riquelme: rikel@google.com
> >
> > George Tucker: gjt@google.com
> >
> > Jasper Snoek: jsnoek@goolge.com

## Motivation

* Exploration and Exploitation

  * 추천 시스템의 주요한 쟁점 중 하나
  * Greedy, E-greedy, UCB, Thompson Sampling의 방법이 있음
  * 이 중 Thompson Sampling 방법론은 확률 분포를 통해 탐색-실행의 비율을 조정
  * 사후 분포(posterior distribution)는 관측된 결과의 누적에 따라 형태가 수렴함
  * 사후 분포에서 sampling한 값을 사용하는 것이 Thompson sampling의 아이디어

  

* Bayesian Linear Model

  * 선형 모델의 높은 해석력이 장점
  * 낮은 적합성은 베이지안 관점으로 해결
  * 회귀 계수를 추정할 때 Likelihood를 Maximize하는 점 추정 방식의 대안으로 회귀 계수의 확률 분포를 가정하는 베이지안 관점을 도입
  * 관측된 결과에 따라 각 설명 변수(context vector)에 대한 평균적인 설명값이 아닌 확률 분포상 샘플링된 값을 적용하여 추정값을 계산함



## Model

### Decision Making with Thompson Sampling

##### Notation

* t: time

* d: Dimension of Context Vector
* X: Context vector
  * X = (t X d) matrix
  * Y를 설명하는 변수 벡터, 매트릭스
* a: Action
  * 추천할 대상, 선택할 대상 중 선택된 것
* r: Reward
  * X에게 a를 추천했을 때 기대되는 Reward
* E[r* - r]
  * 목적함수
  * Regret, 빗나간 예측을 최소화 시키고 싶은 목적함수

### Pseudo Code

```
Algorithm: Thompson Sampling
1: Input: Prior distribution over models, 샘플링 b < B ~[0,1]
2: for time t = 0 ... N
3: 			Observe context X
4: 			Sample model b
5: 			Compute a = BestAction(X,b)
6: 			Select action a and Observe reward r
7: 			Update posterior distribution B (X,a,r)
```

### Bayesian Linear Regression

* Posterior Distribution = Likelihood * Prior
  * Likelihood: f(Y|X,B) -- Y = X.T * B
  * P(B | X, Y) = f(Y|X,B) * p(B | X) [X가 고정되어 있다면 p(B)]
  * p(B)의 밀도함수 g
    * g가 평균 0, 표준편차가 람다의 함수인 가우스 분포라면, posterior 분포 B는 Ridege Regression (B**2 의 정규화)
    * g가 평균 0, 스케일 파라미터가 람다의 함수인 이중지수분포라면, posterior 분포 B는 Lasso Regression (|B|의 정규화)

* Y = X.T * B + E [E ~ N(0,o**2)]
  * Y = Reward
* Joint Distribution of B and o**2
  * o**2: noise, exploration parameter
  * Joint(B,o**2) = p(B|o\*\*2) * p(o\*\*2)
  * o**2 ~ IG(a,b) [Inverse Gamma]
  * B|o**2 ~ N(mu, o\*\*2 X Cov) [Gaussian Distribution]
* Expression
  * Cov = (X.T * X + Lamda *  Eye).Inv
  * Mu = Cov * (Lamda*mu(0) + X.T * Y)
  * a = a(0) + t/2
  * b = b(0) + 1/2(Y.T * Y + mu(0).T * Cov * mu(0) - mu.T * Cov.Inv * mu)
* Hyperparameter
  * Lamda * Eye
  * mu(0)
  * a(0) = b(0) > 1
  * Piror Distribution
    * B|o ** 2 ~ N(0, o**2/Lamda * Eye)

![image-20201019004754808](Model.png)

### Code

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

##### init

```python
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
```

```
mu: X 벡터의 차원에 대한 평균적인 회귀 계수를 의미함
초기 mu: 이에 대해 아무것도 모른다고 가정할 때 평균을 0으로 가정함

cov: X 벡터의 표준편차를 의미함
precision: cov와 inverse 관계에 있으며, 샘플링 분포를 결정할 때 
		cov와 precision 중 어느 것을 사용하느냐에 따라 결과 값이 달라짐
lambda : Shrinkage 정도


Inverse Gamma
* Posterior 분포와 Prior 분포의 Conjugate distribution이 가능한 분포 중 하나
* 에러 Term (E ~ N(0,o**2)) 의 시그마(o**2)는 o**2 ~ IG(a,b) Inverse Gammaba를 따르는 결합 분포
* Y = X.T * B + E에서 X는 고정된 값이므로 E의 사전 분포에 따라 X,Y에 대한 B의 사후 분포가 결정됨
* 따라서 사후 분포 역시 Inverse Gamma를 따르는 분포를 이룰 것임
```

##### action

```python
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
```

```
Round-Robin: action에서는 각 개별 추천 대상에 대한 Reward를 모두 계산하는 과정을 거쳐 가장 Reward가 높은 기대값을 추정함
			이에 따라 시행 횟수마다 후보 arm(추천 대상)을 우선순위 없이 동일하게 반복하기 위해 Round-robin을 통해 data-buffer 상황을 세팅함
			
sigma2_s : o**2, Inverse Gamma 분포에서 샘플링하여 계산한 값
beta_s: 사후 분포에서 샘플링한 회귀 계수 값, Update과정을 통해 계산된 평균과 공분산을 가진 사후 분포에서 샘플링한 회귀 계수

except: covariance가 Poisitive Definite이 아닌 경우 LinaAlgError가 발생하기 때문에 이 경우에 B에 대한 사후 분포를 (0,1)의 가우스 분포로 초기 값으로 세팅

vals : 각 X 벡터의 칼럼에 대한 회귀 계수값을 곱하고 Intercept를 더하여 추정한 값의 배열

argmax(vals) : vals가 극대화되는 추천 대상(arm)을 찾음
```

##### update

```python
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
```

```
Update : np.argmax를 통해 선택된 action(arm,추천대상)과 실제 reward를 비교 후 이를 반영하여 B| Y,X 의 사후 분포를 업데이트함

Y = X.T * B 에서, B = (X.T * X).inverse * (X.T * Y)
여기서 규제 파라미터를 고려하게 되면
B = (X.T * X + 람다 * Eye).inverse * (X.T * Y)

precision_a = X.T * X + 람다 * Eye(I)
cov_a = precision_a.Inverse
Mu_a = precisiion.I * X.T * Y
여기서, Mu_0이 0이기 때문에 0이 되는 항이 존재함 (Expression 식의 1번)

a_post : 감마 분포 상 a 업데이트, a(0) + t/2
b_post : b(0) + b_upd
b_upd : Y.T * Y(Reward의 제곱) - 추정치의 제곱 * [0.5]

* a, b를 통해 시행에 따라 실제 값과 추정치의 값의 차이를 반영하여 회귀 계수 B의 joint Distribution을 업데이트 함
* Beta 샘플링의 분포는 (mu, sigma * cov)를 따르며
* 이 때 sigma는 a,b의 inverse gamma 분포에서 샘플링한 값
```

