import pandas as pd
from noshow_iq.preprocess import load_and_clean, get_features_and_target


def test_load_returns_dataframe():
    df = load_and_clean()
    assert isinstance(df, pd.DataFrame)


def test_no_negative_days():
    df = load_and_clean()
    assert (df["days_in_advance"] >= 0).all()


def test_no_show_is_binary():
    df = load_and_clean()
    assert set(df["no_show"].unique()).issubset({0, 1})


def test_age_range():
    df = load_and_clean()
    assert df["age"].min() >= 0
    assert df["age"].max() <= 115


def test_features_and_target_shape():
    df = load_and_clean()
    X, y = get_features_and_target(df)
    assert X.shape[0] == y.shape[0]


def test_no_show_column_not_in_features():
    df = load_and_clean()
    X, y = get_features_and_target(df)
    assert "no_show" not in X.columns
