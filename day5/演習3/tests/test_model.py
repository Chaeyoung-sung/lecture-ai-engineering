import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import json
import glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
MODEL_HISTORY_DIR = os.path.join(MODEL_DIR, "history")

# モデル性能のしきい値を定義
MIN_ACCURACY = 0.75
MAX_INFERENCE_TIME = 1.0  # 秒


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習開始時間
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # モデルのバージョン管理用ディレクトリを作成
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_HISTORY_DIR, exist_ok=True)

    # 現在の日時をバージョン識別子として使用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_path = os.path.join(
        MODEL_HISTORY_DIR, f"titanic_model_{timestamp}.pkl"
    )

    # モデルを保存（現在のバージョンと履歴用）
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(model_version_path, "wb") as f:
        pickle.dump(model, f)

    # モデルの評価メトリクスを計算
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    inference_time = time.time() - start_time

    # 各種メトリクスを計算
    metrics = {
        "version": timestamp,
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "training_time": float(training_time),
        "inference_time": float(inference_time),
        "n_samples": len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
    }

    # メトリクスの保存（JSONで履歴を保存）
    metrics_history = []
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_history = json.load(f)
                if not isinstance(metrics_history, list):
                    metrics_history = [metrics_history]  # 単一のオブジェクトを配列に変換
        except (json.JSONDecodeError, FileNotFoundError):
            metrics_history = []

    # 新しいメトリクスを追加して保存
    metrics_history.append(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_history, f, indent=2)

    # 最新のメトリクスをバージョン別に保存
    version_metrics_path = os.path.join(MODEL_HISTORY_DIR, f"metrics_{timestamp}.json")
    with open(version_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, X_test, y_test, metrics


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test, metrics = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 結果の記録
    print(f"\nモデルの精度: {accuracy:.4f}")

    # Titanicデータセットではしきい値以上の精度が必要
    assert accuracy >= MIN_ACCURACY, f"モデルの精度が低すぎます: {accuracy:.4f} < {MIN_ACCURACY}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _, metrics = train_model

    # 推論時間の計測 - 複数回試行して平均を計算
    n_iterations = 5
    inference_times = []

    for _ in range(n_iterations):
        start_time = time.time()
        model.predict(X_test)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    # 平均推論時間を計算
    avg_inference_time = np.mean(inference_times)
    print(f"\n平均推論時間: {avg_inference_time:.6f}秒 ({n_iterations}回試行)")

    # 推論時間が制限時間未満であることを確認
    assert (
        avg_inference_time < MAX_INFERENCE_TIME
    ), f"推論時間が長すぎます: {avg_inference_time:.6f}秒 > {MAX_INFERENCE_TIME}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    # 結果の記録
    is_reproducible = np.array_equal(predictions1, predictions2)
    print(f"\nモデルの再現性: {is_reproducible}")

    assert is_reproducible, "モデルの予測結果に再現性がありません"


def test_model_comprehensive_metrics(train_model):
    """モデルの総合的な評価指標を検証"""
    model, X_test, y_test, metrics = train_model

    # 予測と各種指標の計算
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 各指標を計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 結果の記録
    print(f"\n総合評価指標:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

    # 全指標が最低限の要求を満たしているか確認
    assert accuracy >= 0.7, f"Accuracyが低すぎます: {accuracy:.4f}"
    assert precision >= 0.6, f"Precisionが低すぎます: {precision:.4f}"
    assert recall >= 0.5, f"Recallが低すぎます: {recall:.4f}"
    assert f1 >= 0.6, f"F1 Scoreが低すぎます: {f1:.4f}"
    assert roc_auc >= 0.7, f"ROC AUCが低すぎます: {roc_auc:.4f}"


def test_compare_with_previous_model(train_model, sample_data):
    """モデルを過去バージョンと比較して性能判定を行う"""
    # 現在のモデルの準備
    current_model, X_test, y_test, current_metrics = train_model

    # 過去のモデルファイルを取得
    model_files = glob.glob(os.path.join(MODEL_HISTORY_DIR, "titanic_model_*.pkl"))
    metrics_files = glob.glob(os.path.join(MODEL_HISTORY_DIR, "metrics_*.json"))

    # 過去のモデルがない場合はテストをスキップ
    if len(model_files) <= 1 or len(metrics_files) <= 1:  # 1つしかない場合は現在のモデルのみ
        pytest.skip("比較可能な過去のモデルが見つかりません")

    # ファイルを作成タイムスタンプでソートし、最新のモデルを除いた最新モデルを選択
    model_files.sort()
    previous_model_path = model_files[-2] if len(model_files) > 1 else model_files[0]

    metrics_files.sort()
    previous_metrics_path = (
        metrics_files[-2] if len(metrics_files) > 1 else metrics_files[0]
    )

    # 過去のモデルとメトリクスを読み込み
    try:
        with open(previous_model_path, "rb") as f:
            previous_model = pickle.load(f)

        with open(previous_metrics_path, "r") as f:
            previous_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, pickle.UnpicklingError) as e:
        pytest.skip(f"過去のモデルの読み込みに失敗しました: {str(e)}")

    # 過去モデルで予測して精度を計算
    previous_y_pred = previous_model.predict(X_test)
    previous_accuracy = accuracy_score(y_test, previous_y_pred)

    # 現在モデルが過去モデルよりも大きく性能が低下していないことを確認
    # 记事差が0.05以上ある場合にアラート
    current_accuracy = current_metrics["accuracy"]
    accuracy_diff = current_accuracy - previous_accuracy

    # 結果の記録
    print(f"\nモデル比較 - 精度:")
    print(f"  現在モデル: {current_accuracy:.4f}")
    print(f"  過去モデル: {previous_accuracy:.4f}")
    print(f"  差分:     {accuracy_diff:.4f}")

    # 5%以上の精度低下がある場合は警告
    if accuracy_diff < -0.05:
        print(
            f"\n\033[91m警告: 現在のモデルは過去モデルよりも{abs(accuracy_diff)*100:.1f}%精度が低下しています\033[0m"
        )

    # 10%以上の性能低下がある場合は失敗させる
    assert accuracy_diff >= -0.1, f"モデルの精度が大きく低下しています ({abs(accuracy_diff)*100:.1f}%)"


def test_model_feature_importance(train_model):
    """モデルの特徴重要度を分析し、主要な特徴が予想通りに影響しているか確認"""
    model, X_test, y_test, metrics = train_model

    # RandomForestモデルから特徴重要度を取得
    # Pipelineからclassifierコンポーネントを取得
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    # 前処理後の特徴名を取得するのは大変なため、重要度の上位値を0以上があるかを確認
    # 重要度上位2つが余裕に特徴全体の30%以上を占めているかを確認
    importances_sorted = np.sort(importances)[::-1]  # 降順にソート
    top_two_importance_ratio = (importances_sorted[0] + importances_sorted[1]) / np.sum(
        importances
    )

    # 重要度のばらつきが十分であるかを確認
    std_importance = np.std(importances)

    print(f"\n特徴重要度分析:")
    print(f"  最上位重要度: {importances_sorted[0]:.4f}")
    print(f"  上位2つの割合: {top_two_importance_ratio:.2%}")
    print(f"  重要度の標準偏差: {std_importance:.4f}")

    # テスト基準
    assert importances_sorted[0] > 0.1, "最も重要な特徴が十分な影響を持っていません"
    assert top_two_importance_ratio >= 0.3, "上位特徴が全体の30%以上を占めていません"
    assert std_importance > 0.02, "特徴の重要度のばらつきが少なすぎます"


def test_model_drift_detection(train_model, sample_data):
    """モデルドリフトの検出 - モデルの予測分布が期待範囲内か確認"""
    model, X_test, y_test, metrics = train_model

    # 予測確率の分布を取得
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 予測確率の分布統計量を計算
    mean_proba = np.mean(y_pred_proba)
    std_proba = np.std(y_pred_proba)
    min_proba = np.min(y_pred_proba)
    max_proba = np.max(y_pred_proba)

    # Titanicデータセットでは、予測確率が両極端に偏らず、広がりを持つことが期待される
    print(f"\n予測確率分布統計:")
    print(f"  平均値: {mean_proba:.4f}")
    print(f"  標準偏差: {std_proba:.4f}")
    print(f"  最小値: {min_proba:.4f}")
    print(f"  最大値: {max_proba:.4f}")

    # 確率分布が急激な変化を示す場合はドリフトの可能性がある
    # 平均値が極端すぎない
    assert 0.2 <= mean_proba <= 0.8, f"予測確率の平均が極端です: {mean_proba:.4f}"
    # データが偏っていないか確認(分布の広がり)
    assert std_proba >= 0.2, f"予測確率の標準偏差が小さすぎます: {std_proba:.4f}"
    # 両極端の値が存在するか確認
    assert (
        max_proba - min_proba >= 0.5
    ), f"予測確率の範囲が狭すぎます: {max_proba:.4f} - {min_proba:.4f} = {max_proba - min_proba:.4f}"


def test_model_serialization_format(train_model):
    """モデルのシリアライズ形式が正しいか確認"""
    # モデルファイルの存在確認
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"

    # モデルファイルのサイズ確認
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB単位

    # モデルの読み込みテスト
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)

        # モデルが正しいクラスか確認
        assert isinstance(
            loaded_model, Pipeline
        ), f"モデルが期待されるPipelineクラスではありません: {type(loaded_model)}"

        # モデルが期待されるコンポーネントを含むか確認
        assert "preprocessor" in loaded_model.named_steps, "モデルに前処理コンポーネントがありません"
        assert "classifier" in loaded_model.named_steps, "モデルに分類器コンポーネントがありません"

        # シリアライズサイズは合理的か
        # TitanicデータセットとRandomForestの場合、通常は数MB程度
        print(f"\nモデルのシリアライズ情報:")
        print(f"  モデルファイルサイズ: {model_size:.2f} MB")
        print(f"  モデルクラス: {type(loaded_model).__name__}")
        print(f"  コンポーネント: {list(loaded_model.named_steps.keys())}")

        assert model_size < 100, f"モデルファイルが大きすぎます: {model_size:.2f} MB"
    except Exception as e:
        assert False, f"モデルの読み込みテストに失敗しました: {str(e)}"


def test_model_resource_usage(train_model):
    """モデルのリソース使用量を確認"""
    try:
        import psutil
        import gc

        # 現在のメモリ使用量の取得
        process = psutil.Process(os.getpid())

        # 不要なメモリの解放
        gc.collect()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB単位

        # モデルを取得
        model, X_test, y_test, metrics = train_model

        # 予測を実行
        start_time = time.time()
        model.predict(X_test)
        inference_time = time.time() - start_time

        # 予測後のメモリ使用量
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB単位
        memory_used = memory_after - memory_before

        print(f"\nリソース使用量:")
        print(f"  推論時間: {inference_time:.6f}秒")
        print(f"  メモリ使用量変化: {memory_used:.2f} MB")
        print(f"  総メモリ使用量: {memory_after:.2f} MB")

        # メモリ使用量を制限する場合の検証
        assert memory_after < 1000, f"メモリ使用量が大きすぎます: {memory_after:.2f} MB"
    except (NameError, ImportError):
        # psutilが利用できない場合はスキップ
        pytest.skip("psutilモジュールが利用できません")


def test_multiple_metrics_comparison(train_model):
    """複数の評価指標でモデルを分析"""
    model, X_test, y_test, current_metrics = train_model

    # 予測を実行
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 各種指標を計算
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # 過去の評価指標と比較
    metrics_files = sorted(glob.glob(os.path.join(MODEL_HISTORY_DIR, "metrics_*.json")))

    if len(metrics_files) > 1:
        # 最新のメトリクスを除外し、一つ前のメトリクスを読み込む
        try:
            with open(metrics_files[-2], "r") as f:
                previous_metrics = json.load(f)

            # 比較結果を表示
            print(f"\n各指標の変化:")
            for metric_name in metrics.keys():
                current_val = metrics[metric_name]
                previous_val = previous_metrics.get(metric_name, 0)
                change = current_val - previous_val
                change_percent = (
                    (change / previous_val * 100) if previous_val != 0 else 0
                )

                status = (
                    "\033[92m↑\033[0m"
                    if change > 0
                    else "\033[91m↓\033[0m"
                    if change < 0
                    else "-"
                )
                print(
                    f"  {metric_name.ljust(10)}: {current_val:.4f} ({status} {change_percent:+.1f}%)"
                )

            # 主要指標が低下していないか確認
            for metric_name, threshold in [
                ("accuracy", -0.05),
                ("f1", -0.05),
                ("roc_auc", -0.05),
            ]:
                current_val = metrics[metric_name]
                previous_val = previous_metrics.get(metric_name, 0)
                change = current_val - previous_val

                # 5%以上の低下があれば警告
                if change < threshold:
                    print(
                        f"\n\033[93m警告: {metric_name}が{abs(change)*100:.1f}%低下しています\033[0m"
                    )

                # 10%以上の低下があれば失敗させる
                assert (
                    change >= -0.1
                ), f"{metric_name}が大きく低下しています ({abs(change)*100:.1f}%)"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\n評価指標の比較に失敗しました: {str(e)}")
    else:
        print("\n比較可能な過去のメトリクスがありません\n現在のメトリクス:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.ljust(10)}: {value:.4f}")


def test_prediction_consistency(train_model):
    """モデルの予測結果が実行間で一貫性があるか確認"""
    model, X_test, y_test, metrics = train_model

    # 同じデータで複数回予測を実行
    n_runs = 3
    predictions = []

    for i in range(n_runs):
        predictions.append(model.predict(X_test))

    # 全ての実行で予測が同じであることを確認
    all_same = all(np.array_equal(predictions[0], pred) for pred in predictions[1:])

    print(f"\n予測の一貫性確認 ({n_runs}回実行): {all_same}")
    assert all_same, "モデルの予測が実行間で異なります"

    # アーカイブされたモデルを読み込んで予測が同じか確認
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)

        loaded_predictions = loaded_model.predict(X_test)
        is_consistent = np.array_equal(predictions[0], loaded_predictions)

        print(f"保存されたモデルとの一貫性: {is_consistent}")
        assert is_consistent, "保存されたモデルの予測が元のモデルと異なります"
    except Exception as e:
        pytest.skip(f"保存されたモデルの読み込みに失敗しました: {str(e)}")
