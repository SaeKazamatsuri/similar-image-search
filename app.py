import os
import subprocess
import pickle
import numpy as np

from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# 1) Flask アプリケーションのセットアップ
# ----------------------------------------------------------

app = Flask(__name__)

# セキュリティのためのシークレットキー（実運用では秘密にすること）
app.secret_key = "replace_with_a_strong_secret_key"

# アップロード許可する拡張子
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# 一時的に画像を保存するフォルダ (static/tmp に保管して、ブラウザから参照)
UPLOAD_FOLDER = "static/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# キャプションとベクトル情報を含む pickle ファイルのパス
PICKLE_PATH = "caption_vectors.pkl"

# DeepDanbooru の学習済みプロジェクトフォルダ（適切なパスに置いておくこと）
DEEPDANBOORU_PROJECT_PATH = "deepdanbooru-v3-20211112-sgd-e28"


# ----------------------------------------------------------
# 2) ヘルパー関数
# ----------------------------------------------------------


def allowed_file(filename):
    """
    アップロードされたファイルの拡張子が許可リストにあるかどうかをチェックする。
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_tmp_folder():
    """
    UPLOAD_FOLDER (static/tmp) の中身をすべて削除する。
    """
    for fname in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, fname)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            # 削除に失敗しても続行
            print(f"Failed to delete {file_path}: {e}")


# ----------------------------------------------------------
# 3) ルート (GET) ：アップロードフォームの表示
# ----------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    demo_images = os.listdir("static/demo_img")
    return render_template("index.html", demo_images=demo_images)


# ----------------------------------------------------------
# 4) アップロード処理 (POST) ：画像受け取り→DeepDanbooru→類似度計算
# ----------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    """
    POST リクエストで送信された画像ファイルを処理する。
    1) tmp フォルダをクリア
    2) ファイルを保存
    3) DeepDanbooru でキャプションを生成
    4) キャプション読み込み
    5) caption_vectors.pkl を読み込んで類似度計算
    6) 上位5件の結果をテンプレートに渡して表示
    """

    # --- 4-1) tmp フォルダをクリア ---
    clear_tmp_folder()

    # --- 4-2) フォームにファイルが含まれているかチェック ---
    if "file" not in request.files:
        flash("ファイルが選択されていません。もう一度試してください。")
        return redirect(url_for("index"))

    file = request.files["file"]

    # --- 4-3) ファイル名が空ではないかチェック ---
    if file.filename == "":
        flash("ファイルが選択されていません。もう一度試してください。")
        return redirect(url_for("index"))

    # --- 4-4) 拡張子が許可されているかチェック ---
    if file and allowed_file(file.filename):
        # 安全なファイル名に変換して保存
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # キャプションファイルのパス（同じ名前で .txt を想定）
        caption_path = save_path.rsplit(".", 1)[0] + ".txt"

        # --------------------------------------------------
        # 4-5) DeepDanbooru でキャプション生成 (subprocess 実行)
        # --------------------------------------------------
        deepdanbooru_command = [
            "deepdanbooru",
            "evaluate",
            UPLOAD_FOLDER,
            "--project-path",
            DEEPDANBOORU_PROJECT_PATH,
            "--allow-folder",
            "--save-txt",
        ]

        try:
            subprocess.run(deepdanbooru_command, check=True)
        except subprocess.CalledProcessError:
            flash("DeepDanbooru によるタグ生成に失敗しました。")
            return redirect(url_for("index"))

        # --------------------------------------------------
        # 4-6) キャプション (.txt) を読み込む
        # --------------------------------------------------
        if not os.path.exists(caption_path):
            flash("キャプションファイルが生成されませんでした。")
            return redirect(url_for("index"))

        with open(caption_path, "r", encoding="utf-8") as f:
            query_caption = f.read().strip()

        # --------------------------------------------------
        # 4-7) caption_vectors.pkl を読み込み、類似度を計算
        # --------------------------------------------------
        try:
            with open(PICKLE_PATH, "rb") as f:
                file_list, tfidf_matrix, vectorizer = pickle.load(f)
        except AttributeError:
            # custom_tokenizer を解決する必要がある場合の処理例
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == "custom_tokenizer":

                        def custom_tokenizer(text):
                            return text.split(", ")

                        return custom_tokenizer
                    return super().find_class(module, name)

            with open(PICKLE_PATH, "rb") as f2:
                unpickler = CustomUnpickler(f2)
                file_list, tfidf_matrix, vectorizer = unpickler.load()

        # クエリキャプションをベクトル化
        query_vector = vectorizer.transform([query_caption])

        # コサイン類似度の計算
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

        # 自分自身（アップロードした画像）が file_list に含まれる場合は除外しつつ、
        # 類似度の高い順に上位5件を取得する
        top_indices = np.argsort(similarities)[::-1]
        top_results = [
            (file_list[i], similarities[i])
            for i in top_indices
            if file_list[i] != filename
        ][:5]

        # --------------------------------------------------
        # 4-8) テンプレートに結果を渡してレンダリング
        # --------------------------------------------------
        # top_results は [(画像ファイル名, 類似度スコア), …] のリスト
        return render_template("index.html", results=top_results, query_image=filename)

    else:
        flash(
            "許可されていないファイル形式です。PNG, JPG, GIF のいずれかを選択してください。"
        )
        return redirect(url_for("index"))


# ----------------------------------------------------------
# 5) アプリケーションを実行する
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)
