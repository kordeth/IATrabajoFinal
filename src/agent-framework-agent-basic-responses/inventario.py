"""Herramientas locales del agente de inventarios de Laboratorios Roster.

Todo se calcula en la PC, sin internet, a partir de dos archivos CSV:

- data/productos.csv: catálogo con stock disponible, stock en tránsito,
  tiempo de reposición (lead time) y nivel de servicio objetivo.
- data/ventas_mensuales.csv: historial de unidades vendidas por mes y SKU.

IMPORTANTE: los datos incluidos son ILUSTRATIVOS (generados para la práctica).
Para usar datos reales, reemplace los CSV manteniendo las mismas columnas.

Pronóstico: para cada SKU se comparan dos métodos con un backtest de los
últimos 6 meses (MAPE): Holt-Winters aditivo con estacionalidad de 12 meses y
el "método actual" (promedio móvil de 3 meses). Se usa el de menor error.
El punto de reorden usa el error real del backtest como medida de variabilidad.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist, mean
from typing import Annotated

from agent_framework import tool

DATA_DIR = Path(__file__).with_name("data")
TEMPORADA = 12  # meses
DIAS_POR_MES = 30
MESES_BACKTEST = 6
DIAS_SOBRESTOCK = 120  # cobertura mayor a ~4 meses se considera sobrestock
NOTA_DATOS = "Datos ilustrativos de práctica; no son cifras reales de Roster."


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Producto:
    sku: str
    descripcion: str
    categoria: str
    origen: str
    stock_disponible: int
    stock_en_transito: int
    lead_time_dias: int
    nivel_servicio: float
    costo_unitario_soles: float


@lru_cache(maxsize=1)
def cargar_productos() -> dict[str, Producto]:
    with (DATA_DIR / "productos.csv").open(encoding="utf-8", newline="") as f:
        return {
            fila["sku"].strip().upper(): Producto(
                sku=fila["sku"].strip().upper(),
                descripcion=fila["descripcion"].strip(),
                categoria=fila["categoria"].strip(),
                origen=fila["origen"].strip(),
                stock_disponible=int(fila["stock_disponible"]),
                stock_en_transito=int(fila["stock_en_transito"]),
                lead_time_dias=int(fila["lead_time_dias"]),
                nivel_servicio=float(fila["nivel_servicio"]),
                costo_unitario_soles=float(fila["costo_unitario_soles"]),
            )
            for fila in csv.DictReader(f)
        }


@lru_cache(maxsize=1)
def cargar_ventas() -> dict[str, list[tuple[str, int]]]:
    """Devuelve {sku: [(mes 'AAAA-MM', unidades), ...]} ordenado por mes."""
    ventas: dict[str, list[tuple[str, int]]] = {}
    with (DATA_DIR / "ventas_mensuales.csv").open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            sku = fila["sku"].strip().upper()
            ventas.setdefault(sku, []).append((fila["mes"].strip(), int(fila["unidades"])))
    for serie in ventas.values():
        serie.sort()
    return ventas


def _normalizar_sku(sku: str) -> str:
    sku_n = (sku or "").strip().upper()
    productos = cargar_productos()
    if sku_n not in productos:
        disponibles = ", ".join(sorted(productos))
        raise ValueError(f"SKU '{sku}' no existe. SKUs válidos: {disponibles}.")
    return sku_n


def _siguiente_mes(mes: str, n: int = 1) -> str:
    anio, m = int(mes[:4]), int(mes[5:7])
    total = anio * 12 + (m - 1) + n
    return f"{total // 12}-{total % 12 + 1:02d}"


# ---------------------------------------------------------------------------
# Modelos de pronóstico
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AjusteHW:
    alpha: float
    beta: float
    gamma: float
    nivel: float
    tendencia: float
    estacional: tuple[float, ...]  # índice estacional por posición en la serie
    residuos: tuple[float, ...]  # errores de un paso dentro de la muestra


def _ajustar_hw(y: list[float], alpha: float, beta: float, gamma: float) -> AjusteHW:
    n = len(y)
    if n < TEMPORADA + 1:
        raise ValueError("Se necesitan al menos 13 meses de historia para pronosticar.")
    # Inicialización con el primer año: tendencia promedio interanual, nivel al
    # final del primer año y estacionalidad sin el efecto de la tendencia.
    promedio = mean(y[:TEMPORADA])
    centro = (TEMPORADA - 1) / 2
    tendencia = mean(y[i + TEMPORADA] - y[i] for i in range(n - TEMPORADA)) / TEMPORADA
    nivel = promedio + tendencia * centro
    estacional = [v - (promedio + tendencia * (i - centro)) for i, v in enumerate(y[:TEMPORADA])]
    residuos: list[float] = []
    for t in range(TEMPORADA, n):
        s = estacional[t - TEMPORADA]
        prediccion = nivel + tendencia + s
        residuos.append(y[t] - prediccion)
        nivel_ant = nivel
        nivel = alpha * (y[t] - s) + (1 - alpha) * (nivel + tendencia)
        tendencia = beta * (nivel - nivel_ant) + (1 - beta) * tendencia
        estacional.append(gamma * (y[t] - nivel) + (1 - gamma) * s)
    return AjusteHW(alpha, beta, gamma, nivel, tendencia, tuple(estacional), tuple(residuos))


@lru_cache(maxsize=256)
def _mejor_ajuste(y: tuple[float, ...]) -> AjusteHW:
    """Elige alpha, beta y gamma que minimizan el error cuadrático de un paso."""
    mejor: AjusteHW | None = None
    mejor_sse = math.inf
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5):
        for beta in (0.01, 0.05, 0.1):
            for gamma in (0.1, 0.2, 0.3):
                ajuste = _ajustar_hw(list(y), alpha, beta, gamma)
                sse = sum(r * r for r in ajuste.residuos)
                if sse < mejor_sse:
                    mejor, mejor_sse = ajuste, sse
    assert mejor is not None
    return mejor


def pronostico_hw(y: list[float], horizonte: int) -> list[float]:
    ajuste = _mejor_ajuste(tuple(float(v) for v in y))
    n = len(y)
    resultado = []
    for h in range(1, horizonte + 1):
        retroceso = TEMPORADA * math.ceil(h / TEMPORADA)
        s = ajuste.estacional[n + h - 1 - retroceso]
        resultado.append(max(0.0, ajuste.nivel + h * ajuste.tendencia + s))
    return resultado


def pronostico_promedio_movil(y: list[float], horizonte: int, ventana: int = 3) -> list[float]:
    """Método actual de referencia: promedio de los últimos 3 meses."""
    return [mean(y[-ventana:])] * horizonte


METODOS = {
    "Holt-Winters": pronostico_hw,
    "Promedio móvil 3 meses": pronostico_promedio_movil,
}


def _mape(reales: list[float], predichos: list[float]) -> float:
    pares = [(r, p) for r, p in zip(reales, predichos) if r > 0]
    return 100 * mean(abs(r - p) / r for r, p in pares)


@dataclass(frozen=True)
class Evaluacion:
    metodo: str  # método elegido (menor MAPE)
    mape: dict[str, float]  # MAPE % por método
    rmse: dict[str, float]  # RMSE en unidades por método

    def como_dict(self) -> dict:
        base = self.mape["Promedio móvil 3 meses"]
        hw = self.mape["Holt-Winters"]
        return {
            "metodo_elegido": self.metodo,
            "mape_holt_winters_pct": round(hw, 1),
            "mape_promedio_movil_pct": round(base, 1),
            "mejora_holt_winters_vs_promedio_movil_pct": round(100 * (base - hw) / base, 1)
            if base
            else 0.0,
            "explicacion": (
                f"Backtest: pronóstico a un mes para cada uno de los últimos {MESES_BACKTEST} "
                "meses. Se usa el método con menor MAPE. Una mejora negativa significa que "
                "Holt-Winters fue peor que el promedio móvil para este SKU."
            ),
        }


@lru_cache(maxsize=256)
def _evaluar(y: tuple[float, ...]) -> Evaluacion:
    serie = list(y)
    reales = serie[-MESES_BACKTEST:]
    mape, rmse = {}, {}
    for nombre, funcion in METODOS.items():
        predichos = [
            funcion(serie[:t], 1)[0] for t in range(len(serie) - MESES_BACKTEST, len(serie))
        ]
        mape[nombre] = _mape(reales, predichos)
        rmse[nombre] = math.sqrt(mean((r - p) ** 2 for r, p in zip(reales, predichos)))
    elegido = min(mape, key=mape.get)
    return Evaluacion(elegido, mape, rmse)


def evaluar_metodos(y: list[float]) -> Evaluacion:
    """Compara Holt-Winters contra el promedio móvil (método actual) con un backtest."""
    return _evaluar(tuple(float(v) for v in y))


# ---------------------------------------------------------------------------
# Lógica de negocio (funciones puras, fáciles de probar)
# ---------------------------------------------------------------------------


def _serie(sku: str) -> tuple[list[str], list[float]]:
    datos = cargar_ventas().get(sku, [])
    return [m for m, _ in datos], [float(u) for _, u in datos]


def _pronosticar(y: list[float], horizonte: int) -> tuple[list[float], Evaluacion]:
    evaluacion = evaluar_metodos(y)
    return METODOS[evaluacion.metodo](y, horizonte), evaluacion


def resumen_producto(sku: str) -> dict:
    sku = _normalizar_sku(sku)
    p = cargar_productos()[sku]
    meses, y = _serie(sku)
    demanda_diaria = _pronosticar(y, 1)[0][0] / DIAS_POR_MES
    cobertura = p.stock_disponible / demanda_diaria if demanda_diaria else math.inf
    return {
        "sku": p.sku,
        "descripcion": p.descripcion,
        "categoria": p.categoria,
        "origen": p.origen,
        "stock_disponible": p.stock_disponible,
        "stock_en_transito": p.stock_en_transito,
        "lead_time_dias": p.lead_time_dias,
        "ventas_ultimo_mes": {"mes": meses[-1], "unidades": int(y[-1])},
        "cobertura_dias_stock_disponible": round(cobertura),
        "valor_inventario_soles": round(p.stock_disponible * p.costo_unitario_soles, 2),
    }


def pronostico_producto(sku: str, meses: int = 3) -> dict:
    sku = _normalizar_sku(sku)
    meses = max(1, min(int(meses), 12))
    nombres_meses, y = _serie(sku)
    valores, evaluacion = _pronosticar(y, meses)
    return {
        "sku": sku,
        "descripcion": cargar_productos()[sku].descripcion,
        "metodo_usado": evaluacion.metodo,
        "ultimos_6_meses_reales": [
            {"mes": m, "unidades": int(u)} for m, u in zip(nombres_meses[-6:], y[-6:])
        ],
        "pronostico": [
            {"mes": _siguiente_mes(nombres_meses[-1], i + 1), "unidades": round(v)}
            for i, v in enumerate(valores)
        ],
        "total_pronosticado": round(sum(valores)),
        "evaluacion_backtest": evaluacion.como_dict(),
        "nota": NOTA_DATOS,
    }


def reposicion_producto(sku: str) -> dict:
    sku = _normalizar_sku(sku)
    p = cargar_productos()[sku]
    _, y = _serie(sku)
    valores, evaluacion = _pronosticar(y, 1)
    pronostico_mes = valores[0]
    demanda_diaria = pronostico_mes / DIAS_POR_MES

    # La variabilidad se mide con el error real del pronóstico en el backtest.
    rmse_mensual = evaluacion.rmse[evaluacion.metodo]
    sigma_diaria = rmse_mensual / math.sqrt(DIAS_POR_MES)
    z = NormalDist().inv_cdf(p.nivel_servicio)
    stock_seguridad = z * sigma_diaria * math.sqrt(p.lead_time_dias)
    demanda_lead_time = demanda_diaria * p.lead_time_dias
    punto_reorden = demanda_lead_time + stock_seguridad

    posicion = p.stock_disponible + p.stock_en_transito
    cobertura_posicion = posicion / demanda_diaria if demanda_diaria else math.inf

    if p.stock_disponible < demanda_lead_time and posicion < punto_reorden:
        estado = "CRÍTICO: riesgo de quiebre de stock antes de que llegue un nuevo pedido"
    elif posicion <= punto_reorden:
        estado = "REPONER: la posición de inventario está en o bajo el punto de reorden"
    elif cobertura_posicion > DIAS_SOBRESTOCK:
        estado = "SOBRESTOCK: inventario para más de 4 meses, riesgo de obsolescencia"
    else:
        estado = "OK: inventario suficiente"

    cantidad = 0
    if posicion <= punto_reorden:
        objetivo = punto_reorden + pronostico_mes  # cubre lead time + un ciclo mensual
        cantidad = int(math.ceil((objetivo - posicion) / 10.0) * 10)

    return {
        "sku": sku,
        "descripcion": p.descripcion,
        "estado": estado,
        "metodo_pronostico": evaluacion.metodo,
        "demanda_diaria_estimada": round(demanda_diaria, 1),
        "lead_time_dias": p.lead_time_dias,
        "nivel_servicio": p.nivel_servicio,
        "stock_seguridad": round(stock_seguridad),
        "punto_reorden": round(punto_reorden),
        "stock_disponible": p.stock_disponible,
        "stock_en_transito": p.stock_en_transito,
        "posicion_inventario": posicion,
        "cobertura_dias": round(cobertura_posicion),
        "cantidad_sugerida_pedido": cantidad,
        "costo_estimado_pedido_soles": round(cantidad * p.costo_unitario_soles, 2),
        "formula": "ROP = demanda diaria x lead time + Z x sigma diaria x raiz(lead time)",
        "nota": NOTA_DATOS,
    }


def alertas() -> dict:
    resultado = []
    for sku in sorted(cargar_productos()):
        r = reposicion_producto(sku)
        if not r["estado"].startswith("OK"):
            resultado.append(
                {
                    "sku": sku,
                    "descripcion": r["descripcion"],
                    "estado": r["estado"],
                    "cobertura_dias": r["cobertura_dias"],
                    "cantidad_sugerida_pedido": r["cantidad_sugerida_pedido"],
                }
            )
    prioridad = {"C": 0, "R": 1, "S": 2}
    resultado.sort(key=lambda a: prioridad.get(a["estado"][0], 3))
    return {"total_alertas": len(resultado), "alertas": resultado, "nota": NOTA_DATOS}


# ---------------------------------------------------------------------------
# Herramientas expuestas al modelo (function calling)
# ---------------------------------------------------------------------------

SkuParam = Annotated[str, "Código SKU del producto, por ejemplo LC-ESF-DIA"]


def _a_json(resultado: dict) -> str:
    """Entrega el resultado al modelo como JSON legible (con tildes y ñ)."""
    return json.dumps(resultado, ensure_ascii=False)


@tool(result_parser=_a_json)
def listar_productos() -> dict:
    """Lista el catálogo de productos ópticos de Roster con su SKU, stock y lead time."""
    productos = [
        {
            "sku": p.sku,
            "descripcion": p.descripcion,
            "origen": p.origen,
            "stock_disponible": p.stock_disponible,
            "stock_en_transito": p.stock_en_transito,
            "lead_time_dias": p.lead_time_dias,
        }
        for p in cargar_productos().values()
    ]
    return {"productos": productos, "nota": NOTA_DATOS}


@tool(result_parser=_a_json)
def consultar_stock(sku: SkuParam) -> dict:
    """Consulta stock disponible, en tránsito, ventas del último mes y días de cobertura de un SKU."""
    return resumen_producto(sku)


@tool(result_parser=_a_json)
def pronosticar_demanda(
    sku: SkuParam,
    meses: Annotated[int, "Cantidad de meses a pronosticar (1 a 12)"] = 3,
) -> dict:
    """Pronostica la demanda mensual de un SKU; compara Holt-Winters con el promedio móvil y usa el de menor error."""
    return pronostico_producto(sku, meses)


@tool(result_parser=_a_json)
def calcular_reposicion(sku: SkuParam) -> dict:
    """Calcula stock de seguridad, punto de reorden y cantidad sugerida de pedido de un SKU."""
    return reposicion_producto(sku)


@tool(result_parser=_a_json)
def alertas_inventario() -> dict:
    """Revisa todos los SKU y devuelve los que tienen riesgo de quiebre, deben reponerse o tienen sobrestock."""
    return alertas()


HERRAMIENTAS_INVENTARIO = [
    listar_productos,
    consultar_stock,
    pronosticar_demanda,
    calcular_reposicion,
    alertas_inventario,
]
