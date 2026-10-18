# RosterStock: agente de inventarios para Laboratorios Roster (PC2)

Agente de IA que ayuda al área de logística de **Laboratorios Roster S.A.C.** a
anticipar la demanda de productos ópticos y a decidir **cuándo y cuánto pedir**,
para evitar quiebres de stock y sobrestock en productos importados con tiempos de
reposición de 30 a 45 días.

- **Modelo:** local, con Foundry Local (por defecto `phi-4-mini`). No usa la nube.
- **Herramientas locales (Python):** stock, pronóstico de demanda, punto de reorden y alertas.
- **MCP (puntos extra):** Microsoft Learn, para consultas técnicas con fuente citada.

> Los datos de `data/` son **ilustrativos** (generados para la práctica). Para usar
> datos reales, reemplace los CSV respetando las mismas columnas.

## Estructura

```
src/agent-framework-agent-basic-responses/
├── main.py            ← agente: modelo local, instrucciones, herramientas y servidor
├── inventario.py      ← herramientas de inventario y pronóstico (function calling)
├── data/
│   ├── productos.csv          ← catálogo, stock, en tránsito, lead time
│   └── ventas_mensuales.csv   ← 24 meses de ventas por SKU
├── tests/             ← pruebas automáticas
└── .env               ← modelo a usar y activación del MCP
```

## Instalación (PowerShell, una sola vez)

```powershell
winget install Microsoft.FoundryLocal
winget install astral-sh.uv
```

Cierre y vuelva a abrir la terminal. Luego, desde la carpeta del proyecto:

```powershell
cd src\agent-framework-agent-basic-responses
uv sync --frozen
foundry model load phi-4-mini
```

## Configuración (`.env`)

```dotenv
FOUNDRY_LOCAL_MODEL=phi-4-mini
ENABLE_MICROSOFT_LEARN_MCP=true
```

- `FOUNDRY_LOCAL_MODEL` acepta el alias (`phi-4-mini`) o el ID exacto que muestra
  `foundry cache list` (por ejemplo `Phi-4-mini-instruct-generic-gpu:5`). Si pone el
  alias, el agente busca solo el ID correcto que publica Foundry Local.
- `ENABLE_MICROSOFT_LEARN_MCP=false` desactiva el MCP (útil sin internet o si el
  modelo se confunde con muchas herramientas).

## Ejecutar

Presione **F5** en VS Code (Run and Debug → *Debug Local Agent HTTP Server*). Se abre
el **Agent Inspector** en el puerto 8088.

O desde la terminal:

```powershell
cd src\agent-framework-agent-basic-responses
uv run --frozen python main.py
```

## Preguntas para probar

| Funcionalidad | Pregunta de ejemplo |
|---|---|
| Catálogo | ¿Qué productos manejamos? |
| Stock | ¿Cuánto stock hay de lentes de contacto diarios? |
| Pronóstico | Pronostica la demanda de LC-ESF-DIA para los próximos 3 meses |
| Reposición | ¿Cuánto debo pedir de GOT-LUB-15 y por qué? |
| Alertas | ¿Qué productos tienen riesgo de quiebre o sobrestock? |
| MCP (extra) | Según Microsoft Learn, ¿qué es Foundry Local? Cita la fuente |
| Fuera de tema | ¿Quién ganó el mundial? (debe negarse amablemente) |

En el Agent Inspector revise que aparezcan las **llamadas a herramientas**
(`function_call`); eso demuestra que la respuesta viene de los cálculos y no de la
memoria del modelo.

## Cómo calcula

- **Pronóstico:** compara dos métodos con un *backtest* de los últimos 6 meses (MAPE):
  Holt-Winters aditivo con estacionalidad de 12 meses y el promedio móvil de 3 meses
  (método actual). Para cada SKU usa el de menor error y lo informa.
- **Stock de seguridad:** `SS = Z × σ diaria × √(lead time)`, con σ tomada del error
  real del backtest y Z según el nivel de servicio del SKU.
- **Punto de reorden:** `ROP = demanda diaria × lead time + SS`.
- **Estados:** CRÍTICO (se agotará antes de que llegue un pedido), REPONER (posición ≤
  ROP), SOBRESTOCK (más de 120 días de cobertura) u OK.

## Pruebas

```powershell
cd src\agent-framework-agent-basic-responses
uv run --frozen python -m unittest discover -s tests -v
```
