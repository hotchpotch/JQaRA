# JQaRA 評価用コード

JQaRA の評価を行うためのサンプルコードです。

## 評価の実行方法

`main.py` を実行することで、各種モデルの評価を行うことができます。初回の実行前は、必要なライブラリのインストールが必要です。

```
pip install -r requirements.txt
```

### 実行例

実行例は以下です。`-m` オプション以下に、モデルを指定することで評価できます。なお、一回実行した結果のデータは `data/eval_results/runs/` 以下に保存され、このデータを 2 回目からは利用します。データの再利用をしたく無い場合は、`--no_cache` オプションをつけて実行ください。

```
python main.py -m \
    BAAI/bge-m3+all \
    BAAI/bge-m3+colbert \
    bclavie/JaColBERT \
    bm25 \
    cl-nagoya/sup-simcse-ja-base \
    corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1 \
    intfloat/multilingual-e5-large \
    pkshatech/GLuCoSE-base-ja \
    text-embedding-3-small
[INFO] - Load JQaRA: total -> 1667
#    Model                                       NDCG@10        MRR@10      NDCG@100       MRR@100
---  ------------------------------------------  -------------  ----------  -------------  ----------
a    bge-m3+all                                  0.576ᵇᶜᵈᵉᶠᵍʰᶦ  0.818ᵈᵉᵍʰᶦ  0.745ᵇᶜᵈᵉᶠᵍʰᶦ  0.820ᵈᵉᵍʰᶦ
b    bge-m3+colbert                              0.566ᶜᵈᵉᵍʰᶦ    0.810ᵈᵉʰᶦ   0.737ᶜᵈᵉʰᶦ     0.812ᵈᵉʰᶦ
c    JaColBERT                                   0.545ᵈᵉʰᶦ      0.807ᵈᵉʰᶦ   0.727ᵈᵉʰᶦ      0.810ᵈᵉʰᶦ
d    bm25                                        0.458ᵉʰᶦ       0.702ᵉʰᶦ    0.670ᵉʰᶦ       0.706ᵉʰᶦ
e    sup-simcse-ja-base                          0.324          0.541       0.572          0.550
f    cross-encoder-mmarco-mMiniLMv2-L12-H384-v1  0.559ᶜᵈᵉʰᶦ     0.811ᵈᵉʰᶦ   0.732ᵈᵉʰᶦ      0.814ᵈᵉʰᶦ
g    multilingual-e5-large                       0.554ᵈᵉʰᶦ      0.799ᵈᵉʰᶦ   0.731ᵈᵉʰᶦ      0.801ᵈᵉʰᶦ
h    GLuCoSE-base-ja                             0.308          0.518       0.564          0.527
i    text-embedding-3-small                      0.388ᵉʰ        0.611ᵉʰ     0.617ᵉʰ        0.617ᵉʰ
```

### オプション

`-r` で指標メトリクスの指定ができます。内部で検索ランキング評価ライブラリの[ranx](https://github.com/amenra/ranx)を使っているため、[ranx のメトリクス](https://amenra.github.io/ranx/metrics/) の指定が可能です。

例:

```
python main.py -r "map@20,hit_rate@20,precision@20,ndcg@20" -m \
    BAAI/bge-m3+all \
    intfloat/multilingual-e5-large
[INFO] - Load JQaRA: total -> 1667
#    Model                  MAP@20      Hit Rate@20  P@20    NDCG@20
---  ---------------------  --------  -------------  ------  ---------
a    bge-m3+all             0.425ᵇ            0.979  0.274ᵇ  0.602ᵇ
b    multilingual-e5-large  0.402             0.976  0.267   0.582
```

`-f` で出力フォーマットの指定ができます。標準は table です。csv や markdown, markdown_with_links 等々、コピペに便利な出力も指定できます。

```
python main.py -f csv -m \
    BAAI/bge-m3+all \
    intfloat/multilingual-e5-large

[INFO] - Load JQaRA: total -> 1667
model_names,ndcg@10,mrr@10,ndcg@100,mrr@100
bge-m3+all,0.576,0.8178,0.7454,0.8202
multilingual-e5-large,0.554,0.7988,0.731,0.8011
```

`-d` で、件数を 20 件に絞ってデバッグ用に実行することもできます。新しいモデルを試す際などにご利用ください。なお、通常実行した結果が保存され残ってしまうので、とりわけデバッグ時には `--no_cache` をつけてご利用ください。

```
python main.py -d -m bm25 --no_cache
[INFO] - Load JQaRA: total -> 1667
[INFO] - Use 20 samples for debug mode
[DEBUG] - Pick qrels
pick qrels: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 34635.05it/s]
[DEBUG] - Run: bm25
[DEBUG] - - Reranker: <class 'reranker.bm25_reranker.BM25Reranker'>
bm25: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 121.19it/s]
#    Model      NDCG@10    MRR@10    NDCG@100    MRR@100
---  -------  ---------  --------  ----------  ---------
a    bm25         0.438     0.641       0.631      0.644

```

その他のオプションは `-h` をご覧ください

## ライセンス

この評価コードのライセンスは [./LICENSE](./LICENSE) の通りの MIT License となっています。
