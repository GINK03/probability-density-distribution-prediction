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

## 問題設定1. 欠けた部分から、もとの分布を予想する  
ある日のデータが何らの原因で欠けてしまった場合、周りの傾向を学習することで、欠けてしまったログから予想を試みます  

<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/36878980-d1de3932-1e04-11e8-8c16-f52644929c9c.png">
</div>

ところどころ、データが欠損しています。  

これに対してDeepLearningのモデルで欠けた分布の予想をしていきましょう  

<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/36879007-f460eedc-1e04-11e8-839a-887280cba7c0.png">
</div>

このように周辺分布となる非常に細かいところは欠けでしまいました（多分活性化関数の工夫の次第です）が、おおよそ再現できることがわかりました。  

ディープラーニングはサンプリング数が十分に多ければ、真の分布を仮説せずとも、直接、求めることができることがわかりました。

なお、このときに用いた損失関数は、Kullback-Leibler情報量 + 平均誤差の混合した損失関数です  
コードで表すとこんな感じです  
```python
def custom_objective(y_true, y_pred):
  mse = K.mean(K.square(y_true-y_pred), axis=-1)
  y_true_clip = K.clip(y_true, K.epsilon(), 1)
  y_pred_clip = K.clip(y_pred, K.epsilon(), 1)
  kullback_leibler = K.sum(y_true_clip * K.log(y_true_clip / y_pred_clip), axis=-1)
  return mse + kullback_leibler / 1000.0
```
数式で表現すると以下のようになります
TODO
これは、Image to Image[1]の論文と、この発表[2]に参考にしました（有効性の検証は別途必要でしょう）　　

## 問題設定2.　異常値検知を行う  
検定の話ですが、一般的に95%信頼区間に入るかどうかがよく使われる手法です。  

信頼区間は確率密度関数が判明している必要がありますが、複雑な分布を本質的にもつようなものに当て推量で分布を仮定することが、事前に知識を外挿していることと等価であり、やらなくて済むならやらないほうがいいでしょう（何らかの方法でサンプリングする必要があります）。  
<div align="center">
  <img width="400px" src="https://user-images.githubusercontent.com/4949982/36882442-4ef29b90-1e17-11e8-94c4-696f849a40c4.png">
</div>
<div align="center">
  <img width="400px" src="https://user-images.githubusercontent.com/4949982/36883133-6244b530-1e1b-11e8-9b89-5b3997570bfb.png">
</div>
この時、この分布が仮に確率分布と定義できるようなサンプリングをした場合、異常値の検出は、あるサンプルした点から95%の全体の面積を占める範囲外であるとき、と考えられれそうです。（これはディープラーニングの出力値が、離散値なので上から順に足し算、下から順に足し算でかんたんに求まります）

例えば、上記のような端っこの方に①のポイントにサンプルされたような場合、それが異常値かどうかは、②までの積分値（足し算）に、③の全体を割って、0.05以下になるとかと定義することが可能そうです  


## 参考文献
- [1] [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- [2] [確率分布間の距離推定：機械学習分野における最新動向](https://www.jstage.jst.go.jp/article/jsiamt/23/3/23_KJ00008829126/_pdf)
