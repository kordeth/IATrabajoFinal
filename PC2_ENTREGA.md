# PC2: Agente de IA del proyecto final

**Proyecto:** Predicción de demanda con Machine Learning para la gestión de
inventarios de productos ópticos importados de Laboratorios Roster S.A.C.
**Agente:** RosterStock

---

## 1. Avance del proyecto evidenciado con el agente

El proyecto final busca reducir quiebres de stock y sobrestock en los productos
importados de Roster, que tienen tiempos de reposición de 30 a 45 días. El agente
**RosterStock** lleva ese objetivo a una herramienta conversacional. El área de
logística le pregunta en lenguaje natural y el agente:

1. consulta el stock disponible y en tránsito de cada SKU;
2. pronostica la demanda mensual, comparando un modelo de series de tiempo
   (Holt-Winters) contra el método actual (promedio móvil);
3. calcula stock de seguridad, punto de reorden y cantidad sugerida de pedido;
4. genera alertas priorizadas: riesgo de quiebre, reponer y sobrestock.

Los cálculos los hace código Python (las herramientas del agente), no el modelo de
lenguaje. El modelo entiende la pregunta, elige la herramienta y explica el resultado.
Así se evita que invente cifras.

## 2. Modelo seleccionado y justificación

**Modelo:** `phi-4-mini` (Microsoft), ejecutado **en local** con Foundry Local.

| Criterio | Por qué phi-4-mini |
|---|---|
| Privacidad | Stock, costos y ventas son datos comerciales sensibles. Con un modelo local no salen de la PC. |
| Herramientas | `foundry model info phi-4-mini` muestra la capacidad *tool-calling*. Es indispensable para que el agente llame a las funciones de inventario. |
| Contexto | 131 072 tokens, suficiente para el historial de ventas y las instrucciones. |
| Licencia | MIT: uso académico y comercial sin costo. |
| Costo | Sin pago por uso ni suscripción a la nube. |
| Hardware | Tiene variante para GPU y para CPU, así que corre en las laptops del grupo. |

**Alternativas evaluadas:**

- `gemma-4-e2b-it`: también corre en local, pero pesa 6.9 GB (GPU) o 5.9 GB (CPU).
- `phi-4-mini-reasoning`: es mejor para razonamiento matemático, pero el agente no
  lo necesita porque los cálculos los hace Python. Lo importante aquí es que use
  bien las herramientas.
- *Opcional, nube:* un modelo grande en Azure AI Foundry daría respuestas más
  fluidas, pero implica costo y enviar datos comerciales fuera de la empresa.

## 3. Instrucciones del agente (system prompt)

Están en `main.py`, en la variable `INSTRUCTIONS`. En resumen:

- Rol: asistente de planificación de inventarios de Roster.
- Objetivo: evitar quiebres de stock y sobrestock.
- Responder en español, breve y con cifras concretas.
- Usar **siempre** las herramientas para stock, pronósticos y pedidos, y nunca
  inventar cifras.
- Al recomendar un pedido, indicar SKU, cantidad, punto de reorden y motivo.
- Indicar el método de pronóstico y su error (MAPE).
- Usar Microsoft Learn (MCP) solo para dudas técnicas, citando la URL.
- Rechazar con amabilidad los temas ajenos a inventarios.

## 4. Funcionalidades demostradas

| Herramienta | Qué hace | Ejemplo con los datos de práctica |
|---|---|---|
| `listar_productos` | Catálogo de 6 SKU | Lentes esféricos, diarios, tóricos, de color, solución y gotas |
| `consultar_stock` | Stock, tránsito y cobertura | LC-ESF-DIA: 380 cajas, 18 días de cobertura |
| `pronosticar_demanda` | Pronóstico y backtest | SOL-MUL-360: Holt-Winters MAPE 4.6 % vs promedio móvil 11.1 % |
| `calcular_reposicion` | SS, ROP y pedido | LC-ESF-DIA: ROP 1 030, pedir 1 270 cajas (S/ 66 040) |
| `alertas_inventario` | Alertas priorizadas | 2 críticos (LC-ESF-DIA, GOT-LUB-15) y 1 sobrestock (LC-COL-MEN) |
| **MCP Microsoft Learn** (extra) | Documentación oficial con fuente | "¿Qué es Foundry Local?" con la URL citada |

**Resultado honesto del backtest:** con los datos de práctica, Holt-Winters gana en
SOL-MUL-360, que tiene la estacionalidad más marcada. En los otros 5 SKU gana el
promedio móvil, porque 24 meses con ruido son poca historia para estimar bien la
estacionalidad. Por eso el agente elige el mejor método **por SKU** en vez de forzar
uno solo. Con 36 meses o más de datos reales se espera que el modelo estacional gane
en más productos. Esto conecta con el siguiente paso del proyecto: probar XGBoost y
LSTM con la historia real.

## 5. Guion del video (1 minuto)

| Tiempo | En pantalla | Qué decir |
|---|---|---|
| 0:00–0:08 | VS Code con `main.py` | "Este es RosterStock, el agente de nuestro proyecto de predicción de demanda para Laboratorios Roster." |
| 0:08–0:15 | `.env` y terminal con `foundry model list` | "Usa phi-4-mini en local con Foundry Local: los datos no salen de la PC y soporta llamadas a herramientas." |
| 0:15–0:22 | `INSTRUCTIONS` en `main.py` | "Sus instrucciones: responder en español, usar siempre las herramientas y no inventar cifras." |
| 0:22–0:32 | Agent Inspector: "¿Qué productos tienen riesgo de quiebre?" | "Detecta dos productos críticos y uno con sobrestock." |
| 0:32–0:42 | "¿Cuánto debo pedir de LC-ESF-DIA y por qué?" | "Calcula punto de reorden y cantidad sugerida. Aquí se ve la llamada a la herramienta." |
| 0:42–0:50 | "Pronostica la demanda de SOL-MUL-360 para 3 meses" | "Compara Holt-Winters con el promedio móvil y usa el de menor error." |
| 0:50–1:00 | "Según Microsoft Learn, ¿qué es Foundry Local?" | "Y como extra, consulta documentación oficial por MCP y cita la fuente." |

**Consejo:** antes de grabar, haga cada pregunta una vez para que el modelo ya esté
cargado. En el Agent Inspector, despliegue las llamadas a herramientas para que se
vean en el video.
