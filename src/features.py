import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_DATOS_LIMPIOS = "data/processed/dataset_limpio.csv"
RUTA_TRAIN = "data/processed/train.csv"
RUTA_TEST = "data/processed/test.csv"

ANIO_REFERENCIA = 2026
COLS_NUMERICAS = ["Antiguedad", "Motor", "Kilometraje"]
COLS_CATEGORICAS = ["Marca", "Transmisión", "Combustible"]
COL_OBJETIVO = "Precio"


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables derivadas necesarias antes del modelado."""
    df_out = df.copy()
    df_out["Antiguedad"] = ANIO_REFERENCIA - df_out["Año"]
    return df_out


def construir_preprocesador() -> ColumnTransformer:
    """
    Construye el ColumnTransformer sin ajustarlo aún (evita fuga de datos).
    - Numéricas: imputación por mediana + estandarización (clave para Ridge/Lasso).
    - Categóricas: imputación por moda + One-Hot Encoding evitando multicolinealidad.
    """
    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", pipeline_numerico, COLS_NUMERICAS),
        ("cat", pipeline_categorico, COLS_CATEGORICAS),
    ])


def generar_particiones(
    ruta_entrada: str = RUTA_DATOS_LIMPIOS,
    ruta_train: str = RUTA_TRAIN,
    ruta_test: str = RUTA_TEST,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Carga los datos limpios, selecciona variables y guarda los splits train/test."""
    df = pd.read_csv(ruta_entrada)
    df = crear_variables(df)

    columnas_finales = COLS_NUMERICAS + COLS_CATEGORICAS + [COL_OBJETIVO]
    df_modelo = df[columnas_finales].copy()

    train_df, test_df = train_test_split(
        df_modelo,
        test_size=test_size,
        random_state=random_state,
    )

    os.makedirs(os.path.dirname(ruta_train), exist_ok=True)
    train_df.to_csv(ruta_train, index=False, encoding="utf-8")
    test_df.to_csv(ruta_test, index=False, encoding="utf-8")

    print(f"Variables listas: {COLS_NUMERICAS + COLS_CATEGORICAS}")
    print(f"Entrenamiento guardado en {ruta_train}: {len(train_df)} filas")
    print(f"Prueba guardado en {ruta_test}: {len(test_df)} filas")
    return train_df, test_df


if __name__ == "__main__":
    generar_particiones()