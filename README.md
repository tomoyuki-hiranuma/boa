# BOA  
Bayesian Optimization Algorithm(BOA)の実装
## BOAとは  
分布推定アルゴリズムの一種で分布を表す確率モデルにベイジアンネットワークを用いた最適化手法である．  
### アルゴリズム  
1: 初期集団生成P(0)  
2: 集団P(t)から上位評価値集団S(t)を選択  
3: 上位評価値集団S(t)からBN構築  
4: BNから集団O(t)生成  
5: P(t)の評価値の低い個体群をO(t)と交換することでP(t+1)を生成  
6: 終了条件満たしていなければ2へ  

### ベイジアンネットワークの探索方法
- Scoring metric : BicScore  
- Searching method : HillClimbMethod  
- 探索したモデルのCPDは最尤推定で求める  