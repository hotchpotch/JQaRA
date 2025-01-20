# **JQaRA** : Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語 Q&A データセット

高性能な LLM の台頭に伴い、LLM を用いた質疑応答のユースケースが増加しています。しかしながら、LLM は質問に対して適切な回答する知識を有していないと、答えることができないだけでなく、誤った回答を返答するといった課題が存在します。この課題を解決する一つの手段として、LLM が外部の知識を参照して回答する「RAG（Retrieval-Augmented Generation・検索拡張生成）」の需要が高まっています。

そのため、LLM が RAG を用いた際に回答精度が上がるような情報を検索によって取得可能か評価するためのデータセット"**JQaRA** : Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語 Q&A データセット"を構築しました。なお JQaRA は「じゃくら」と読みます。

データセット自体は HuggingFace で、データセットの評価コード例などは GitHub で公開しています。

- 🤗 https://huggingface.co/datasets/hotchpotch/JQaRA
  - HuggingFace で公開している JQaRA データセットです
- 🛠️ https://github.com/hotchpotch/JQaRA/
  - GitHub で、📈 [評価用コード](https://github.com/hotchpotch/JQaRA/tree/main/evaluator) を公開しています。

## JQaRA の特徴

JQaRA の特徴として、llama-7b の派生モデルや GPT4 等の LLM が質問に回答できる検索データに対して正解ラベル付けを行っています(注・一部人間の目視チェックよるラベル付もあり)。そのため、LLM にとって回答精度を上げるヒントになるデータをどれだけ検索で取得できるか、すなわち RAG の精度向上に寄与しそうかの視点を元に作ったデータセットです。

大元の質問文は[AI 王 公式配布データセット(JAQKET)](https://sites.google.com/view/project-aio/dataset?authuser=0)を、検索対象文は Wikipedia のデータを用いています。

### 評価指標

JQaRA は質問に対して、候補となる 100 件のデータ(一件以上の正解を含む)の情報検索タスクです。そのため主の評価指標として、test データの nDCG@10 (normalized Documented Cumulative Gain)を用います。

また例として、簡単に評価できるスクリプトを [GitHub の evaluator](https://github.com/hotchpotch/JQaRA/tree/main/evaluator) 以下に置いています。このスクリプトは SentenceTransformer や CrossEncoder といった、一般的なインターフェイスを備えたモデル、また高精度と謳われるモデルを評価するスクリプトです。

### 評価結果

以下はさまざまなモデルの評価結果です。評価は nDCG@10 以外にも、参考まで MRR@10 の数値も掲載しています。

#### 密な文ベクトルモデル

| model_names                                                                     | ndcg@10 | mrr@10 |
| :------------------------------------------------------------------------------ | ------: | -----: |
| [bge-m3+dense](https://huggingface.co/BAAI/bge-m3)                              |   0.539 | 0.7854 |
| [fio-base-japanese-v0.1](https://huggingface.co/bclavie/fio-base-japanese-v0.1) |  0.3718 | 0.6161 |
| [sup-simcse-ja-base](https://huggingface.co/cl-nagoya/sup-simcse-ja-base)       |  0.3237 | 0.5407 |
| [sup-simcse-ja-large](https://huggingface.co/cl-nagoya/sup-simcse-ja-large)     |  0.3571 |  0.575 |
| [unsup-simcse-ja-base](https://huggingface.co/cl-nagoya/unsup-simcse-ja-base)   |  0.3121 | 0.5209 |
| [unsup-simcse-ja-large](https://huggingface.co/cl-nagoya/unsup-simcse-ja-large) |  0.3928 | 0.6257 |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)  |   0.554 | 0.7988 |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)  |  0.4917 | 0.7291 |
| [GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja)             |  0.3085 | 0.5179 |
| [GLuCoSE-base-ja-v2](https://huggingface.co/pkshatech/GLuCoSE-base-ja-v2)   |    0.606  |   0.8359 |
| [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)    |  0.3881 | 0.6107 |
| [ruri-large](https://huggingface.co/cl-nagoya/ruri-large)                   |    0.6287 |   0.8418 |
| [ruri-base](https://huggingface.co/cl-nagoya/ruri-base)                     |    0.5833 |   0.8093 |
| [ruri-small](https://huggingface.co/cl-nagoya/ruri-small)                   |    0.5359 |   0.7661 |
| [static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese) |    0.4704 |   0.6814 |


#### ColBERT モデル

| model_names                                               | ndcg@10 | mrr@10 |
| :-------------------------------------------------------- | ------: | -----: |
| [bge-m3+colbert](https://huggingface.co/BAAI/bge-m3)      |  0.5656 | 0.8095 |
| [JaColBERT](https://huggingface.co/bclavie/JaColBERT)             |    0.5488 |   0.8116 |
| [JaColBERTv2](https://huggingface.co/bclavie/JaColBERTv2)         |    0.5906 |   0.8316 |
| [JaColBERTv2.4](https://huggingface.co/answerdotai/JaColBERTv2.4) |    0.6265 |   0.8556 |
| [JaColBERTv2.5](https://huggingface.co/answerdotai/JaColBERTv2.5) |    0.642  |   0.8647 |



#### CrossEncoder モデル

| model_names                                                                                                             | ndcg@10 | mrr@10 |
| :---------------------------------------------------------------------------------------------------------------------- | ------: | -----: |
| [japanese-reranker-cross-encoder-xsmall-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-xsmall-v1) |    0.6136 |   0.8402 |
| [japanese-reranker-cross-encoder-small-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-small-v1)   |    0.6247 |   0.8599 |
| [japanese-reranker-cross-encoder-base-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-base-v1)     |    0.6711 |   0.8809 |
| [japanese-reranker-cross-encoder-large-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1)   |    0.71   |   0.8983 |
| [japanese-bge-reranker-v2-m3-v1](https://huggingface.co/hotchpotch/japanese-bge-reranker-v2-m3-v1)                       |    0.6918 |   0.8996 |
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)                                                     |    0.673  |   0.8909 |
| [shioriha-large-reranker](https://huggingface.co/cl-nagoya/shioriha-large-reranker)                                      |    0.5775 |   0.83   |
| [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                                      |  0.2445 | 0.4378 |
| [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                                    |  0.4718 | 0.7108 |
| [cross-encoder-mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1) |  0.5588 | 0.8107 |
| [ruri-reranker-small](https://huggingface.co/cl-nagoya/ruri-reranker-small) |    0.6453 |   0.8637 |
| [ruri-reranker-base](https://huggingface.co/cl-nagoya/ruri-reranker-base)   |    0.7429 |   0.9113 |
| [ruri-reranker-large](https://huggingface.co/cl-nagoya/ruri-reranker-large) |    0.7712 |   0.9098 |

#### スパースベクトルモデル

| model_names                                         | ndcg@10 | mrr@10 |
| :-------------------------------------------------- | ------: | -----: |
| [japanese-splade-base-v1](https://huggingface.co/hotchpotch/japanese-splade-base-v1) |    0.6441 |   0.8616 |
| [bge-m3+sparse](https://huggingface.co/BAAI/bge-m3) |  0.5088 | 0.7596 |
| bm25                                                |   0.458 |  0.702 |

#### その他モデル

| model_names                                         | ndcg@10 | mrr@10 |
| :-------------------------------------------------- | ------: | -----: |
| [bge-m3+all](https://huggingface.co/BAAI/bge-m3)    |   0.576 | 0.8178 |

---

## JQaRA データセット構築方法

### Q＆A データの選定

まず JQaRA の基礎となる日本語 Q&A データとして、「[JAQKET: クイズを題材にした日本語 QA データセット](https://sites.google.com/view/project-aio/dataset?authuser=0)」の質問と回答を使用しています。JAQKET は、質の高い多様な日本語 Q&A データセットで、Wikipedia の記事タイトル名が回答となる特徴を持っています。そのため、Wikipedia の文章から適切な該当文章を見つけることができれば、ほとんどの質問に対して回答を導き出すことが可能です。

JAQKET の中から、CC-BY-SA 4.0 ライセンスで公開されている dev(約 2,000 件)、unused(約 600 件)、test(約 2,000 件)のデータを JQaRA で使用しています。JAQKET の train(約 17000 件)はライセンスが学術利用用途のみとなり、商用での学習は不可なことから含めていません。以下は、JQaRA 評価用の test データセットの構築方法です。

### Wikipedia データの追加

JAQKET の質問データから、Wikipedia から質問に関連するであろう文章を取得します。Wikipedia の記事全文だと文章が長すぎるため、最大文字数が 400 文字になるようにチャンク分割されたデータ、[singletongue/wikipedia-utils - passages-c400-jawiki-20230403](https://huggingface.co/datasets/singletongue/wikipedia-utils)を利用しています。

質問文から関連する文章の取得には、Embeddings モデルを用いた文ベクトルの類似度で評価視しています。また一つの Embeddings モデルでは偏りが発生してしまうため、多様性を確保するために 5 種類の Embeddings モデル[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [cl-nagoya/sup-simcse-ja-base](https://huggingface.co/cl-nagoya/sup-simcse-ja-base), [pkshatech/GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja), [OpenAI/text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) を利用します。

また 400 文字以内になるように分割された Wikipedia 文データは、約 560 万文存在します。そのため、現実的な速度で検索が可能になるよう、Embeddings モデルを用いて文ベクトルに変換した後、IVF(Inverted File Index)と量子化(IVFPQ)を使い、高速にベクトル検索が可能な状態にします。なおベクトル検索のライブラリには FAISS を用いており、IVFPQ のパラメータは IVF の nlist に 2048、PQ は Embeddings モデルの埋め込みベクトルの次元数/4(例: e5-large は 1024 次元なので、PQ=1024/4=256)としています。

これらを使い、質問文各々に最も類似する上位 500 の文章 x 5 種類の Embeddings モデルの結果を得ます。その後、これら 5 つの結果を RRF(Reciprocal Rank Fusion)を用いてランク付けしなおし、スコアが高い上位 100 文を抽出しました。これらの文と、その文が含まれる Wikipedia 記事タイトルを、質問文に紐付けします。

### ルールベースでの正解ラベルの付与

質問文に紐付けした 100 文の中から、タイトルまたは文に質問に対応する回答文字列が完全一致で含まれる場合には、それを関連があると判断し正解ラベルをまず付与します。質問文に紐づけた 100 文のうち、正解ラベルが一つもないデータ(39 件)や、正解ラベルの数が多すぎるデータは、評価が困難なため除外しました。正解ラベルの数が多いデータの算出には、各質問に紐づく正解ラベルの総数の標準偏差を計算し、総数平均値(16.54) +1 標準偏差(15.21) = 31.66 件以上の正解ラベルを持つデータ 281 件を除外しました。

このフィルタリングにより、元々約 2,000 件あった test データが 1,680 件へと減少しました。また、正解ラベルが付与されている文データは、1,680 質問 \* 100 文の合計 168,000 データのうち 16,726 件となりました。また各々の質問について、100 文中何件正解ラベルの総数は、フィルタリング後は平均 9.98 件、標準偏差は 6.70 となっています。

## 正解ラベル有用性の検証

ルールベースで付与された正解ラベルの中には、質問に対して回答精度を上げるのヒントとならないものも含まれています。そのためまず実際の LLM を用いて、質問と文(Wikipedia タイトル + 400 文字以下の対象文)を与えた上で、正しい回答が出力できるかどうかで、その正解ラベルが有益かどうかを評価します。最終的には人間での評価も行い、回答精度を上げるヒントにならないデータは削除します。

### 1) LocalLLM 7B, 13B モデルでの検証

初めに、日本語 LLM のパラメータ数が 7B の [youri-7b-instruction](https://huggingface.co/rinna/youri-7b-instruction) および、13B の [Swallow-13B-instruction-hf](https://huggingface.co/tokyotech-llm/Swallow-13b-instruct-hf) に対して回答のみを出力するように SFT(Supervised fine-tuning)で学習させたモデル、[youri-7b-stf-qa-context-jaqket-jsquad-gptq](https://huggingface.co/hotchpotch/youri-7b-stf-qa-context-jaqket-jsquad-gptq)と [Swallow-13b-stf-qa-context-jaqket-jsquad-gptq](https://huggingface.co/hotchpotch/Swallow-13b-stf-qa-context-jaqket-jsquad-gptq) を作成しました。なお、今回の test データは、これらのモデルの学習には利用していません。

この段階で、ルールベースの正解ラベルが付与されたデータ 16,726 件中、どちらかのモデルが部分一致で正解を含む回答を出せなかった物が、16,726 件中 742 件ありました。なお部分一致なのは、回答生成時に少々冗長な出力をしがちなため、完全一致ではなく部分一致で評価しています。

### 2) ChatGPT 3.5, GPT4 での検証

その後の二段階目では、1) の 7B, 13B モデルで間違えたデータ 742 件を使用し、ChatGPT 3.5(gpt-3.5-turbo-0125)および GPT4(gpt-4-0125-preview)を用いて、同様に正解が出せるかを検証しました。この結果、ChatGPT 3.5, GPT4 どちらのモデルも間違ったデータは 550 件ありました。

### 3) 人間の検証

最後に、LLM が間違えたデータ 550 件を人間がチェックして、正解ラベルを削除するかどうかを判断し、最終的に 522 件を削除するデータとみなしました。

このうち、正解ラベルとして削除しない、と判断したデータで一番多いものは表記揺れです。LLM 評価が文字列一致だったため、例えば質問が"「ぼたん」と言えばイノシシの肉ですが、「もみじ」といえば何の肉でしょう?"で、正解は"シカ"ですが、LLM が"鹿肉"と答えたようなものが挙げられます。

また、質問が "炊飯器、冷蔵庫、洗濯機といった生活に必須となる家電製品のことを、ある色を用いた表現で何というでしょう?" で、正解は"白物家電"ですが、文の中の表記が"ホワイトグッズ(白物家電)"となっていて、LLM は皆"ホワイトグッズ"を回答として挙げていました。

他にも、大抵の人なら読んで正解を答えられるであろうが、LLM では間違ったしまったものも残す判断をしました。例えば、質問が"約 1.8 リットルを 1 とする、日本酒などの体積を表す時に用いる尺貫法の単位は何でしょう?" に対して、文として "斗(と)とは、尺貫法における体積(容積)の単位。10 升が 1 斗、10 斗が 1 石となる。日本では、明治時代に 1 升=約 1.8039 リットルと定められたので、1 斗=約 18.039 リットルとなる。" が与えられているので、正解の「升」を答えられるが、LLM 達は「斗」と誤って回答したデータ等も削除せずに残しています。

## test データセットの構築

最後に、522 件のデータの正解ラベルをラベルを削除します。ラベルのみの削除なので、データ自体は残しています。また正解ラベルの削除により、1,680 件の質問に紐づいている 100 文の中で、正解ラベルが 1 つもないデータが発生するので、それを除いた最終的な質問データは 1,667 件となりました。このデータで test データセットは構築されています。

## dev, unused データセットの作成・構築

dev, unused データセットについても、test の文データと重複が発生しないよう取り除いて、ほぼ同様の方法で作成しました。ただし、dev, unused データセットに置いては、正解ラベル有用性の検証では youri-7b のモデルのみを用いて正しい回答が出力されたデータ最大 5 件を残し、他のルールベースで正解をつけたが LLM が答えられなかったデータはラベルの削除ではなく、データ自体を削除しています。また、dev, unused では 1 質問に対して 100 文ではなく 50 文を付与しています。

最終的に、dev 1,737 件、unused 498 件の質問データで、データセットを作りました。なお本来、学習には train のデータを利用しますが、大元の Q&A データセット JAQKET train データがライセンス上商用利用できないため、JQaRA ではこの dev, unused のデータセットを学習用途として想定しております。

# おわりに

今回、JQaRA データセットを構築しようと思ったのは、実務や実験において RAG の精度がどれだけ上がるかの評価が、既存のデータセットを用いたことでは難しかったことから、無いなら自ら作ってみようと始まりました。趣味で作り始めたこともあり、途中で挫折せずに公開まで行き着くことができて嬉しく思います。

これまで、特に自然言語処理の分野では、研究、コミュニティ、企業からのアウトプットに多大な恩恵を受けてきました。このデータセットが、自然言語処理や検索技術に携わる方々に少しでも貢献できれば幸いです。

# ライセンス

JQaRA データセットのライセンスは、"question", "answers" カラムは[AI 王 公式配布データセット(JAQKET)](https://sites.google.com/view/project-aio/dataset?authuser=0)  の[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)を継承します。

また "title", "text" カラムにおいては、[Wikipedia の著作権である CC BY-SA 4.0 または GFDL](https://ja.wikipedia.org/wiki/Wikipedia:%E8%91%97%E4%BD%9C%E6%A8%A9)とします。

## 謝辞

このデータセットは、[AI 王 公式配布データセット(JAQKET)](https://sites.google.com/view/project-aio/dataset?authuser=0)から質問・回答文を利用しています。AI 王の開催・クイズ作成等々をされた関係者の方々、有益なデータセットの公開ありがとうございます。

また関連文章を見つけるために利用した、有益な Embeddings モデルを公開されている大学・研究機関・企業の方々、ありがとうございます。

---

```
@misc{yuichi-tateno-2024-jqara,,
    url={https://huggingface.co/datasets/hotchpotch/JQaRA},
    title={JQaRA: Japanese Question Answering with Retrieval Augmentation - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
    author={Yuichi Tateno}
}
```
