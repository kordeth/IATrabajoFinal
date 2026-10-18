import unittest

import inventario as inv


class InventarioTests(unittest.TestCase):
    def test_catalogo_y_ventas_cargan(self):
        productos = inv.cargar_productos()
        ventas = inv.cargar_ventas()
        self.assertEqual(len(productos), 6)
        for sku in productos:
            self.assertGreaterEqual(len(ventas[sku]), 24)

    def test_sku_invalido_da_error_claro(self):
        with self.assertRaisesRegex(ValueError, "SKUs válidos"):
            inv.consultar_stock.func("NO-EXISTE") if hasattr(inv.consultar_stock, "func") else inv.resumen_producto("NO-EXISTE")

    def test_sku_no_distingue_mayusculas(self):
        self.assertEqual(inv.resumen_producto(" lc-esf-dia ")["sku"], "LC-ESF-DIA")

    def test_holt_winters_reproduce_serie_estacional_exacta(self):
        patron = [100, 90, 120, 100, 95, 98, 130, 102, 95, 97, 100, 140]
        serie = [v + 2 * i for i, v in enumerate(patron * 3)]
        pronostico = inv.pronostico_hw(serie, 12)
        esperado = [v + 2 * (36 + i) for i, v in enumerate(patron)]
        for p, e in zip(pronostico, esperado):
            self.assertAlmostEqual(p, e, delta=1.0)

    def test_pronostico_tiene_meses_y_metodo_elegido_por_menor_mape(self):
        r = inv.pronostico_producto("SOL-MUL-360", 3)
        self.assertEqual([p["mes"] for p in r["pronostico"]], ["2026-10", "2026-11", "2026-12"])
        ev = r["evaluacion_backtest"]
        menor = "Holt-Winters" if ev["mape_holt_winters_pct"] <= ev["mape_promedio_movil_pct"] else "Promedio móvil 3 meses"
        self.assertEqual(r["metodo_usado"], menor)

    def test_meses_se_limita_entre_1_y_12(self):
        self.assertEqual(len(inv.pronostico_producto("GOT-LUB-15", 50)["pronostico"]), 12)
        self.assertEqual(len(inv.pronostico_producto("GOT-LUB-15", 0)["pronostico"]), 1)

    def test_reposicion_es_coherente(self):
        for sku in inv.cargar_productos():
            r = inv.reposicion_producto(sku)
            self.assertGreaterEqual(r["stock_seguridad"], 0)
            self.assertGreater(r["punto_reorden"], 0)
            if r["posicion_inventario"] <= r["punto_reorden"]:
                self.assertGreater(r["cantidad_sugerida_pedido"], 0)
                self.assertEqual(r["cantidad_sugerida_pedido"] % 10, 0)
            else:
                self.assertEqual(r["cantidad_sugerida_pedido"], 0)

    def test_alertas_ordenadas_por_prioridad(self):
        estados = [a["estado"] for a in inv.alertas()["alertas"]]
        self.assertTrue(estados)
        orden = [{"C": 0, "R": 1, "S": 2}[e[0]] for e in estados]
        self.assertEqual(orden, sorted(orden))

    def test_herramientas_tienen_esquema_para_el_modelo(self):
        nombres = [t.name for t in inv.HERRAMIENTAS_INVENTARIO]
        self.assertEqual(
            nombres,
            ["listar_productos", "consultar_stock", "pronosticar_demanda", "calcular_reposicion", "alertas_inventario"],
        )
        params = inv.pronosticar_demanda.parameters()
        self.assertIn("sku", params["properties"])
        self.assertEqual(params["required"], ["sku"])


if __name__ == "__main__":
    unittest.main()
