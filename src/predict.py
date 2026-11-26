import joblib
import pandas as pd

try:
    from src.features import ANIO_REFERENCIA
    from src.train import RUTA_MEJOR_MODELO
except ImportError:
    from features import ANIO_REFERENCIA
    from train import RUTA_MEJOR_MODELO

MARCAS_VALIDAS = [
    "Chevrolet", "Honda", "Hyundai", "Kia",
    "Mazda", "Mitsubishi", "Nissan", "Toyota",
]
TRANSMISIONES_VALIDAS = ["Automático", "Estándar"]
COMBUSTIBLES_VALIDOS = ["Gasolina", "Diésel"]


def predecir_precio(
    marca: str,
    anio: int,
    motor: float,
    kilometraje: float,
    transmision: str,
    combustible: str,
    ruta_modelo: str = RUTA_MEJOR_MODELO,
) -> float:
    """Devuelve el precio estimado en USD para un vehículo usado."""
    pipeline = joblib.load(ruta_modelo)

    datos_vehiculo = pd.DataFrame([{
        "Antiguedad": ANIO_REFERENCIA - int(anio),
        "Motor": float(motor),
        "Kilometraje": float(kilometraje),
        "Marca": marca.strip(),
        "Transmisión": transmision.strip(),
        "Combustible": combustible.strip(),
    }])

    prediccion = float(pipeline.predict(datos_vehiculo)[0])
    return round(max(1000.0, prediccion), 2)


def pedir_opcion(etiqueta: str, opciones: list) -> str:
    """Solicita una opción válida al usuario en consola."""
    mapa = {op.lower(): op for op in opciones}
    while True:
        valor = input(f"{etiqueta} {opciones}: ").strip()
        if valor.lower() in mapa:
            return mapa[valor.lower()]
        print(f"Opción no válida. Elige entre: {', '.join(opciones)}")


def main():
    print("=== Cotizador de Vehículos Usados — El Salvador 2026 ===")
    marca = pedir_opcion("Marca", MARCAS_VALIDAS)
    anio = int(input("Año del vehículo (ej. 2020): ").strip())
    motor = float(input("Motor en litros (ej. 1.8): ").strip())
    kilometraje = float(input("Kilometraje recorrido (ej. 85000): ").strip())
    transmision = pedir_opcion("Transmisión", TRANSMISIONES_VALIDAS)
    combustible = pedir_opcion("Combustible", COMBUSTIBLES_VALIDOS)

    precio_estimado = predecir_precio(
        marca=marca,
        anio=anio,
        motor=motor,
        kilometraje=kilometraje,
        transmision=transmision,
        combustible=combustible,
    )

    print(f"\nPrecio estimado de mercado: ${precio_estimado:,.2f} USD")


if __name__ == "__main__":
    main()