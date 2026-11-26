import os
import sys
import unittest
import numpy as np
import pandas as pd

# Permite importar desde src/ al ejecutar desde la raíz del repositorio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    COL_OBJETIVO,
    COLS_CATEGORICAS,
    COLS_NUMERICAS,
    RUTA_DATOS_LIMPIOS,
    RUTA_TEST,
    RUTA_TRAIN,
    construir_preprocesador,
    crear_variables,
)
from src.predict import predecir_precio
from src.train import RUTA_MEJOR_MODELO


class TestPipelineAutos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_limpio = pd.read_csv(RUTA_DATOS_LIMPIOS)
        cls.train_df = pd.read_csv(RUTA_TRAIN)
        cls.test_df = pd.read_csv(RUTA_TEST)

    def test_dataset_limpio_sin_nulos_ni_duplicados(self):
        self.assertEqual(int(self.df_limpio.isnull().sum().sum()), 0)
        self.assertEqual(int(self.df_limpio.duplicated().sum()), 0)

    def test_rangos_logicos_variables(self):
        self.assertTrue((self.df_limpio["Precio"] > 0).all())
        self.assertTrue(self.df_limpio["Año"].between(2000, 2026).all())
        self.assertTrue(self.df_limpio["Motor"].between(0.8, 5.8).all())
        self.assertTrue((self.df_limpio["Kilometraje"] > 0).all())

    def test_crear_variables_antiguedad(self):
        df_feat = crear_variables(self.df_limpio)
        self.assertIn("Antiguedad", df_feat.columns)
        self.assertTrue((df_feat["Antiguedad"] >= 0).all())

    def test_preprocesador_forma_y_valores_finitos(self):
        cols_x = COLS_NUMERICAS + COLS_CATEGORICAS
        X_train = self.train_df[cols_x]
        X_test = self.test_df[cols_x]

        preprocesador = construir_preprocesador()
        X_train_t = preprocesador.fit_transform(X_train)
        X_test_t = preprocesador.transform(X_test)

        # 3 numéricas + 7 marcas (8-1) + 1 transmisión + 1 combustible = 12 columnas
        self.assertEqual(X_train_t.shape[1], 12)
        self.assertEqual(X_test_t.shape[1], 12)
        self.assertFalse(np.isnan(X_train_t).any())
        self.assertFalse(np.isinf(X_train_t).any())
        self.assertFalse(np.isnan(X_test_t).any())
        self.assertFalse(np.isinf(X_test_t).any())

    def test_prediccion_modelo_serializado(self):
        self.assertTrue(os.path.exists(RUTA_MEJOR_MODELO), "Primero ejecuta src/train.py")
        precio = predecir_precio(
            marca="Toyota",
            anio=2020,
            motor=1.8,
            kilometraje=85000,
            transmision="Automático",
            combustible="Gasolina",
        )
        self.assertIsInstance(precio, float)
        self.assertGreater(precio, 1000.0)


if __name__ == "__main__":
    unittest.main()