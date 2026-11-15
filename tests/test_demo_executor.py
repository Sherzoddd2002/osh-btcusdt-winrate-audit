from __future__ import annotations

import os
import unittest
from decimal import Decimal
import httpx
from unittest.mock import patch

from demo_executor.engine import DemoEngine

from demo_executor.client import BinanceDemoClient, DEMO_FUTURES_URL
from demo_executor.logic import (
    barrier_prices, choose_side, decimal_floor_step, quantity_for_cap,
    round_price_to_tick,
)


class DemoLogicTests(unittest.TestCase):
    def test_only_demo_url_is_configured(self):
        self.assertEqual(DEMO_FUTURES_URL, "https://testnet.binancefuture.com")
        self.assertNotIn("https://fapi.binance.com", DEMO_FUTURES_URL)
        self.assertNotIn("https://demo-fapi.binance.com", DEMO_FUTURES_URL)

    def test_multi_asset_mode_reads_account_setting(self):
        client = BinanceDemoClient("testnet-key", "testnet-secret")
        with patch.object(client, "_request", return_value={"multiAssetsMargin": True}) as request:
            result = client.multi_asset_mode()
        self.assertTrue(result["multiAssetsMargin"])
        request.assert_called_once_with("GET", "/fapi/v1/multiAssetsMargin", signed=True)
        client.close()

    def test_conditional_stop_uses_algo_endpoint(self):
        client = BinanceDemoClient("demo-key", "demo-secret")
        with patch.object(client, "_request", return_value={"algoStatus": "NEW"}) as request:
            client.place_algo_order({
                "algoType": "CONDITIONAL", "symbol": "BTCUSDT",
                "side": "SELL", "type": "STOP_MARKET",
                "triggerPrice": "60000", "quantity": "0.001",
                "reduceOnly": True, "clientAlgoId": "OSHSL123",
            })
        request.assert_called_once_with(
            "POST", "/fapi/v1/algoOrder",
            {
                "algoType": "CONDITIONAL", "symbol": "BTCUSDT",
                "side": "SELL", "type": "STOP_MARKET",
                "triggerPrice": "60000", "quantity": "0.001",
                "reduceOnly": True, "clientAlgoId": "OSHSL123",
            },
            signed=True,
        )
        client.close()

    def test_network_error_preserves_safe_root_cause(self):
        client = BinanceDemoClient("demo-key", "demo-secret")
        with patch.object(client.http, "request", side_effect=httpx.ConnectError("DNS lookup failed")):
            with self.assertRaisesRegex(RuntimeError, "ConnectError.*DNS lookup failed"):
                client._request("GET", "/fapi/v1/time")
        client.close()
    def test_generic_binance_environment_keys_are_not_used_as_demo_keys(self):
        with patch.object(DemoEngine, "_read_keyring_credentials", return_value=None):
            with patch.dict(os.environ, {"BINANCE_API_KEY": "possibly-live", "BINANCE_API_SECRET": "possibly-live-secret"}):
                engine = DemoEngine()
        self.assertFalse(engine.credentials_configured)
        self.assertIsNone(engine.client)

    def test_save_credentials_uses_secure_credential_store_without_returning_secret(self):
        class FakeKeyring:
            def __init__(self):
                self.priority = 1
                self.values = {}
            def set_password(self, service, username, password):
                self.values[(service, username)] = password
            def get_password(self, service, username):
                return self.values.get((service, username))
            def delete_password(self, service, username):
                self.values.pop((service, username), None)

        with patch.object(DemoEngine, "_read_keyring_credentials", return_value=None):
            engine = DemoEngine()
        store = FakeKeyring()
        with patch.object(DemoEngine, "_keyring_module", return_value=store):
            result = engine.save_credentials("demo-key", "demo-secret")
        self.assertTrue(result["saved"])
        self.assertEqual(result["storage"], "Windows Credential Manager")
        self.assertNotIn("demo-secret", repr(result))
        self.assertEqual(store.get_password("OSH_Quant_Lab_Binance_Demo", "api_secret"), "demo-secret")
        engine.client.close()

    def test_start_refuses_multi_asset_mode_before_margin_or_order_changes(self):
        engine = DemoEngine()
        engine.orders_enabled = True
        with patch.object(engine, "_load_model", return_value={}):
            with patch.object(engine, "_connect", return_value={"hedge_mode": False, "multi_assets_mode": True}):
                with self.assertRaisesRegex(RuntimeError, "requires Single-Asset Mode"):
                    engine.start(confirm_demo_orders=True)
        self.assertFalse(engine.accept_entries)
        self.assertFalse(engine.thread)

    def test_start_requires_explicit_demo_order_confirmation(self):
        engine = DemoEngine()
        engine.orders_enabled = False
        with self.assertRaisesRegex(RuntimeError, "Explicit Testnet order confirmation"):
            engine.start()

    def test_reset_refuses_to_remove_credentials_when_bot_tracks_position(self):
        engine = DemoEngine()
        engine.credentials_configured = True
        engine.client = object()
        engine.state["trade"] = {"side": "LONG", "quantity": 0.001}
        with self.assertRaisesRegex(RuntimeError, "position exists"):
            engine.reset_credentials()
        self.assertFalse(engine.accept_entries)
        self.assertTrue(engine.credentials_configured)
        self.assertIsNotNone(engine.client)

    def test_algo_order_status_uses_client_algo_id(self):
        client = BinanceDemoClient("demo-key", "demo-secret")
        with patch.object(client, "_request", return_value={"algoStatus": "NEW"}) as request:
            client.algo_order("BTCUSDT", "OSHSL123")
        request.assert_called_once_with(
            "GET", "/fapi/v1/algoOrder",
            {"symbol": "BTCUSDT", "clientAlgoId": "OSHSL123"},
            signed=True,
        )
        client.close()

    def test_exchange_info_uses_documented_unfiltered_endpoint(self):
        client = BinanceDemoClient("demo-key", "demo-secret")
        with patch.object(client, "_request", return_value={"symbols": []}) as request:
            client.exchange_info("BTCUSDT")
        request.assert_called_once_with("GET", "/fapi/v1/exchangeInfo")
        client.close()

    def test_open_algo_orders_uses_symbol_filter_only(self):
        client = BinanceDemoClient("demo-key", "demo-secret")
        with patch.object(client, "_request", return_value=[]) as request:
            client.open_algo_orders("BTCUSDT")
        request.assert_called_once_with(
            "GET", "/fapi/v1/openAlgoOrders", {"symbol": "BTCUSDT"}, signed=True
        )
        client.close()

    def test_thresholds_control_signals(self):
        self.assertIsNone(choose_side(.5, .5, .8, .8))
        self.assertEqual(choose_side(.85, .4, .8, .8), "LONG")
        self.assertEqual(choose_side(.4, .9, .8, .8), "SHORT")

    def test_dual_signal_chooses_greater_normalized_excess(self):
        self.assertEqual(choose_side(.90, .88, .80, .70), "SHORT")

    def test_h96_c2_barriers(self):
        self.assertAlmostEqual(barrier_prices("LONG", 100000, .02)[0], 101500)
        self.assertAlmostEqual(barrier_prices("LONG", 100000, .02)[1], 98000)
        self.assertAlmostEqual(barrier_prices("SHORT", 100000, .02)[0], 98500)
        self.assertAlmostEqual(barrier_prices("SHORT", 100000, .02)[1], 102000)

    def test_quantity_rounds_down(self):
        q = quantity_for_cap(100, 100000, "0.001", "0.001", 50)
        self.assertEqual(q, Decimal("0.001"))
        self.assertLessEqual(float(q) * 100000, 100.000001)

    def test_never_increases_cap_to_exchange_minimum(self):
        with self.assertRaisesRegex(ValueError, "minimum quantity"):
            quantity_for_cap(50, 100000, "0.001", "0.001", 50)

    def test_min_notional_enforced(self):
        with self.assertRaisesRegex(ValueError, "minimum notional"):
            quantity_for_cap(50, 10000, "0.0001", "0.0001", 100)

    def test_decimal_tick_rounding(self):
        self.assertEqual(round_price_to_tick(100.011, "0.01", "down"), Decimal("100.01"))
        self.assertEqual(round_price_to_tick(100.011, "0.01", "up"), Decimal("100.02"))

    def test_floor_step(self):
        self.assertEqual(decimal_floor_step("0.00199", "0.001"), Decimal("0.001"))


if __name__ == "__main__":
    unittest.main()
