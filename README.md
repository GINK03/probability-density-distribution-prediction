# Deep Learningによる無限次元分布推定

## 例えばこのような分布がある
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/36842371-d2817dfc-1d8e-11e8-9d65-040503024b4b.png">
</div>

## 十分サンプリングでき、かつ、連続な入力に対して予想したい場合  
例えば、この分布が日付のような連続なものとして扱われる場合、ある日のデータがサンプルできなかったり、まだサンプルが住んでいない未来に対して予想しようとした場合、そういうことは可能なのでしょうか。  

古典的なベイズでも可能ですが、ディープラーニングを用いて、KL距離、mean squre error距離などの距離関数で損失を決定して、分布を仮定せず（ビッグデータ流儀）、直接高次元の密度関数を予想可能であることを示したいと思います  

## 例えばこのような分布を仮定
正規分布２つから導から導かれる頻度  
```python
import numpy as np

SIZE = 25000
for i in range(SIZE):
  sample1 = np.random.normal(loc=-2*np.sin(i/10), scale=1, size=10000)
  sample2 = np.random.normal(loc=5*np.cos(i/10), scale=2, size=30000)
```
微妙に距離がある２つの分布から構成されており、i(時系列)でlocのパラメータが変化し、支配的な正規分布と非支配的な分布が入りまじります    

## 問題設定 1.
ある日のデータが何らの原因で欠けてしまった場合、周りの傾向を学習することで、欠けてしまったログから予想を試みます  

<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/36847342-573dfaea-1da1-11e8-8267-96de15743e7b.png">
</div>

ところどころ、データが欠損しています。  

これに対してDeepLearningのモデルで欠けた分布の予想をしていきましょう  
