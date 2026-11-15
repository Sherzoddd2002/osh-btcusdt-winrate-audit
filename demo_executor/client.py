from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode

import httpx

# Fixed non-production Futures Testnet/Demo-compatible REST host.
# The current Demo hostname (demo-fapi.binance.com) is TLS-reset on some networks;
# this legacy Futures Testnet hostname currently serves public /fapi endpoints.
# No environment override and no production/live fallback are permitted.
DEMO_FUTURES_URL = "https://testnet.binancefuture.com"


class BinanceAPIError(RuntimeError):
    def __init__(self, code, message, status_code=None):
        self.code, self.status_code = code, status_code
        super().__init__(f"Binance Demo API error {code}: {message}")


class BinanceDemoClient:
    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("Demo API credentials are missing from the local .env.")
        self.api_key, self.api_secret = api_key.strip(), api_secret.strip()
        self.http = httpx.Client(
            timeout=httpx.Timeout(12.0, connect=6.0),
            headers={"X-MBX-APIKEY": self.api_key},
            follow_redirects=False,
        )
        self.offset_ms = 0
        self.synced = False

    def close(self):
        self.http.close()

    def _request(self, method: str, path: str, params=None, signed=False):
        values = {k: str(v).lower() if isinstance(v, bool) else str(v)
                  for k, v in (params or {}).items() if v is not None}
        if signed:
            if not self.synced and path != "/fapi/v1/time":
                self.sync_time()
            values["timestamp"] = str(int(time.time() * 1000) + self.offset_ms)
            values["recvWindow"] = "5000"
        query = urlencode(values)
        if signed:
            signature = hmac.new(
                self.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            query = f"{query}&signature={signature}"
        url = f"{DEMO_FUTURES_URL}{path}" + (f"?{query}" if query else "")
        try:
            response = self.http.request(method.upper(), url)
        except httpx.HTTPError as exc:
            cause = exc
            while getattr(cause, "__cause__", None) is not None:
                cause = cause.__cause__
            detail = str(cause).strip().replace(chr(13), " ").replace(chr(10), " ")
            if not detail:
                detail = "no additional OS detail"
            detail = detail[:180]
            raise RuntimeError(
                f"Demo API network error: {type(exc).__name__}; "
                f"root cause: {type(cause).__name__}: {detail}"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {"msg": response.text[:300]}
        if response.status_code >= 400:
            raise BinanceAPIError(
                payload.get("code") if isinstance(payload, dict) else None,
                payload.get("msg", "Request failed") if isinstance(payload, dict) else "Request failed",
                response.status_code,
            )
        return payload

    def sync_time(self):
        server_ms = int(self._request("GET", "/fapi/v1/time")["serverTime"])
        self.offset_ms = server_ms - int(time.time() * 1000)
        self.synced = True
        return self.offset_ms

    def exchange_info(self, symbol=None):
        # USD-M exchangeInfo is the full public symbol/filter catalogue.
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, symbol, limit=500):
        return self._request("GET", "/fapi/v1/klines",
                             {"symbol": symbol, "interval": "15m", "limit": limit})

    def ticker_price(self, symbol):
        return float(self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})["price"])

    def account(self):
        return self._request("GET", "/fapi/v3/account", signed=True)

    def positions(self, symbol):
        result = self._request("GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True)
        return result if isinstance(result, list) else [result]

    def position_mode(self):
        return self._request("GET", "/fapi/v1/positionSide/dual", signed=True)

    def multi_asset_mode(self):
        """Return the account-wide USDⓈ-M Multi-Assets/Single-Asset mode."""
        return self._request("GET", "/fapi/v1/multiAssetsMargin", signed=True)

    def open_orders(self, symbol):
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def order(self, symbol, client_id):
        return self._request("GET", "/fapi/v1/order",
                             {"symbol": symbol, "origClientOrderId": client_id}, signed=True)

    def place_order(self, params):
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_order(self, symbol, client_id):
        return self._request("DELETE", "/fapi/v1/order",
                             {"symbol": symbol, "origClientOrderId": client_id}, signed=True)

    # USD-M conditional orders were migrated to Algo endpoints from 2025-12-09.
    def place_algo_order(self, params):
        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def algo_order(self, symbol, client_algo_id):
        return self._request(
            "GET", "/fapi/v1/algoOrder",
            {"symbol": symbol, "clientAlgoId": client_algo_id}, signed=True
        )

    def open_algo_orders(self, symbol):
        result = self._request(
            "GET", "/fapi/v1/openAlgoOrders",
            {"symbol": symbol}, signed=True
        )
        return result if isinstance(result, list) else [result]

    def cancel_algo_order(self, symbol, client_algo_id):
        return self._request(
            "DELETE", "/fapi/v1/algoOrder",
            {"symbol": symbol, "clientAlgoId": client_algo_id}, signed=True
        )

    def set_isolated(self, symbol):
        try:
            return self._request("POST", "/fapi/v1/marginType",
                                 {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except BinanceAPIError as exc:
            if exc.code == -4046:
                return {"msg": "Already isolated"}
            raise

    def set_leverage(self, symbol, leverage):
        return self._request("POST", "/fapi/v1/leverage",
                             {"symbol": symbol, "leverage": leverage}, signed=True)
