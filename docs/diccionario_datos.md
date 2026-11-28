# Diccionario de Datos — Vehículos Usados El Salvador (2026)

## Origen y tratamiento del dataset
- **Archivo crudo:** `data/raw/dataset_autos_sv.csv` (400 anuncios reales de vehículos usados en El Salvador).
- **Archivo limpio:** `data/processed/dataset_limpio.csv` (372 registros válidos, 0 valores nulos, 0 duplicados).
- **Reglas de limpieza aplicadas en `src/clean_data.py`:**
  1. Eliminación de duplicados exactos (antes y después de estandarizar texto).
  2. Conversión de `Precio` desde texto (`$13,700.00`) a flotante y eliminación de precios de relleno (`< $1,000`).
  3. Filtrado de vehículos clásicos anteriores al año 2000 y filas sin año registrado.
  4. Corrección de 19 kilometrajes ingresados en escala abreviada (ej. `130` → `130,000`) e imputación de 3 valores faltantes con la mediana por año.
  5. Extracción numérica de cilindrada (`Motor`) mediante expresiones regulares, conversión de cc a litros y mapeo de códigos de motor (`5vz`, `12r`, `td27`, `2kd`); imputación de valores `"Consultar"` por mediana de `Modelo`/`Marca`.
  6. Exclusión de 2 registros atípicos verificados: Toyota Hilux 2015 a `$5,000` (posible prima) y Hyundai Accent 2001 con `1,462 km`.

## Descripción de columnas

| Variable | Tipo | Uso en el modelo | Rango / Categorías (dataset limpio) | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `Marca` | Categórica nominal | Entrada (`OneHotEncoder`) | 8 marcas (`Nissan`, `Kia`, `Toyota`, `Mitsubishi`, `Hyundai`, `Honda`, `Chevrolet`, `Mazda`) | Fabricante del vehículo. |
| `Modelo` | Categórica nominal | Solo referencia / EDA | 89 modelos únicos | Excluida del modelo final para evitar sobreajuste por alta cardinalidad. |
| `Año` | Entero (`int64`) | Base para `Antiguedad` | `2000` – `2026` (mediana `2019`) | Año de fabricación; transformada en `Antiguedad = 2026 - Año`. |
| `Motor` | Numérica (`float64`) | Entrada (`StandardScaler`) | `1.0` – `5.6` L (mediana `2.0` L) | Cilindrada del motor en litros. |
| `Kilometraje` | Numérica (`float64`) | Entrada (`StandardScaler`) | `1,000` – `414,382` km (mediana `91,158` km) | Recorrido acumulado reportado en el anuncio. |
| `Transmisión` | Categórica binaria | Entrada (`OneHotEncoder`) | `Automático` (330), `Estándar` (42) | Tipo de caja de cambios. |
| `Combustible` | Categórica binaria | Entrada (`OneHotEncoder`) | `Gasolina` (352), `Diésel` (20) | Tipo de combustible utilizado. |
| `Precio` | Numérica (`float64`) | **Variable objetivo (`y`)** | `$2,300` – `$68,000` USD (mediana `$10,850`) | Precio de venta publicado en dólares estadounidenses. |