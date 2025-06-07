## Similar Image Search

このリポジトリは、DeepDanbooru のタグ生成結果を用いて、画像間の類似度検索を行う Web アプリケーションです。

### 概要

1. `DeepDanbooru` プロジェクトを利用して、アップロードされた画像からタグ（キャプション）を生成します。
2. 生成されたキャプションを TF-IDF ベクトル化し、既存の画像コレクションとのコサイン類似度を計算します。
3. 類似度の高い上位 5 件の画像をブラウザに表示します。

### フォルダ構成

```
/ (プロジェクトルート)
├── app.py                     # Flask アプリケーション本体
├── imagePreprocessing.py      # 画像キャプションの TF-IDF 前処理と pickle ファイル生成スクリプト
├── caption_vectors.pkl        # 前処理実行後に生成される pickle ファイル
├── static/
│   ├── demo_img/              # デモ用画像フォルダ（.png/.jpg/.gif）
│   └── tmp/                   # アップロード画像の一時保存フォルダ
└── templates/
    └── index.html             # アップロードフォームおよび検索結果表示用テンプレート
```

### 前提条件

* Python 3.8 以上

### 利用方法

1. リポジトリをクローンします。

   ```bash
   git clone https://github.com/KichangKim/DeepDanbooru.git
   cd DeepDanbooru
   ```

2. DeepDanbooru のサブモジュールとして `similar-image-search` をクローンします。

   ```bash
   git clone https://github.com/SaeKazamatsuri/similar-image-search.git
   ```

3. 必要な Python パッケージをインストールします。

   ```bash
   pip install flask sklearn
   ```

4. 事前にキャプションを用意したい画像フォルダを `similiar-image-search/img/` に配置し、前処理を実行して `caption_vectors.pkl` を生成します。

   ```bash
   cd similar-image-search
   python imagePreprocessing.py
   ```

5. Web アプリケーションを起動します。

   ```bash
   cd ..
   python app.py
   ```

6. ブラウザで以下の URL にアクセスします。

- [http://127.0.0.1:5002/](http://127.0.0.1:5002/)


7. 画像をアップロードすると、上位 5 件の類似画像が表示されます。


### 補足

- `app.py` が参照する `DEEPDANBOORU_PROJECT_PATH` は、実際にダウンロードした DeepDanbooru モデルのパスに書き換えてください。
- `caption_vectors.pkl` を再生成する際は、前処理スクリプトを再実行してください。
- セキュリティのため、`app.secret_key` は環境変数などで管理することを推奨します。

```
