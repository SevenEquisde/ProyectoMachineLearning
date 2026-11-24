import os
import re
import numpy as np
import pandas as pd

RUTA_RAW = "data/raw/dataset_autos_sv.csv"
RUTA_PROCESSED = "data/processed/dataset_limpio.csv"


def limpiar_motor(valor):
    """Extrae la cilindrada en litros (float) a partir de textos variados."""
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()
    if texto.lower() == "consultar":
        return np.nan

    # Códigos de motor reales presentes en el dataset
    codigos_motor = {
        "5vz": 3.4,   # Toyota 5VZ-FE
        "12r": 1.6,   # Toyota 12R
        "td27": 2.7,  # Nissan TD27
        "2kd": 2.5,   # Toyota 2KD-FTV
    }
    if texto.lower() in codigos_motor:
        return codigos_motor[texto.lower()]

    # Corregir errores de escritura: "1..6", "1,6", "2 0"
    texto = texto.replace("..", ".").replace(",", ".")
    if re.fullmatch(r"\d\s+\d", texto):
        texto = re.sub(r"\s+", ".", texto)

    # Buscar primero un número decimal (ej. 2.5 en "2.5L 4c" o 3.8 en "VQ38 - 3.8L")
    match_dec = re.search(r"(\d+\.\d+)", texto)
    if match_dec:
        numero = float(match_dec.group(1))
    else:
        match_int = re.search(r"(\d+)", texto)
        if not match_int:
            return np.nan
        numero = float(match_int.group(1))

    # Convertir centímetros cúbicos a litros (ej. 1600 -> 1.6, 2800 -> 2.8)
    if numero >= 800:
        numero = numero / 1000.0

    # Descartar valores fuera de rango lógico (ej. "6" de 6 cilindros)
    if numero < 0.8 or numero > 5.8:
        return np.nan

    return round(numero, 1)


def limpiar_datos(ruta_entrada=RUTA_RAW, ruta_salida=RUTA_PROCESSED):
    df = pd.read_csv(ruta_entrada)
    filas_iniciales = len(df)

    # 1. Eliminar duplicados exactos (quita 8 filas, incluyendo el Hilux repetido)
    df = df.drop_duplicates().copy()

    # 2. Limpiar espacios y unificar mayúsculas en variables categóricas
    for col in ["Marca", "Modelo", "Transmisión", "Combustible"]:
        df[col] = df[col].astype(str).str.strip()
    df["Modelo"] = df["Modelo"].replace({"RIO": "Rio", "SOUL": "Soul"})

    # 3. Convertir Precio de texto ("$13,700.00") a float y quitar precios falsos ($1)
    df["Precio"] = (
        df["Precio"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )
    df = df[df["Precio"] >= 1000].copy()

    # 4. Filtrar Año válido (sin nulos y vehículos del 2000 al 2026)
    df = df.dropna(subset=["Año"]).copy()
    df["Año"] = df["Año"].astype(int)
    df = df[(df["Año"] >= 2000) & (df["Año"] <= 2026)].copy()

    # 5. Corregir Kilometraje: valores de relleno (1 o >= 999,999) y escala abreviada
    df.loc[(df["Kilometraje"] <= 1) | (df["Kilometraje"] >= 999999), "Kilometraje"] = np.nan
    mask_miles = (df["Año"] < 2025) & (df["Kilometraje"] >= 10) & (df["Kilometraje"] <= 500)
    df.loc[mask_miles, "Kilometraje"] = df.loc[mask_miles, "Kilometraje"] * 1000

    # 6. Limpiar y convertir Motor a numérico (litros)
    df["Motor"] = df["Motor"].apply(limpiar_motor)

    # 7. Imputar nulos restantes (3 en Kilometraje y 55 en Motor)
    mediana_km_anio = df.groupby("Año")["Kilometraje"].transform("median")
    df["Kilometraje"] = df["Kilometraje"].fillna(mediana_km_anio).fillna(df["Kilometraje"].median())

    df["Motor"] = df["Motor"].fillna(df.groupby("Modelo")["Motor"].transform("median"))
    df["Motor"] = df["Motor"].fillna(df.groupby("Marca")["Motor"].transform("median"))
    df["Motor"] = df["Motor"].fillna(df["Motor"].median())
    df["Motor"] = df["Motor"].round(1)

    # Guardar dataset limpio
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df = df.drop_duplicates().copy()
    # Eliminar los 2 registros atípicos acordados por el equipo
    df = df[~((df["Modelo"] == "Hilux") & (df["Año"] == 2015) & (df["Precio"] == 5000))].copy()
    df = df[~((df["Modelo"] == "Accent") & (df["Año"] == 2001) & (df["Kilometraje"] == 1462))].copy()
    df.to_csv(ruta_salida, index=False, encoding="utf-8")

    print(f"Limpieza completada: {filas_iniciales} -> {len(df)} filas guardadas en {ruta_salida}")
    print(f"Nulos restantes: {int(df.isnull().sum().sum())}")
    return df


if __name__ == "__main__":
    limpiar_datos()