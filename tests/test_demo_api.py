from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from demo_executor.app import SESSION, app

PAGES_ORIGIN = "https://sherzoddd2002.github.io"


class DemoApiOriginTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(
            app,
            base_url="http://127.0.0.1:8765",
            client=("127.0.0.1", 54321),
        )

    def tearDown(self):
        self.client.close()

    def test_session_token_is_released_only_to_owned_pages_origin(self):
        good = self.client.get("/api/session", headers={"Origin": PAGES_ORIGIN})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.json()["session"], SESSION)
        self.assertEqual(good.headers.get("access-control-allow-origin"), PAGES_ORIGIN)

        bad = self.client.get("/api/session", headers={"Origin": "https://attacker.example"})
        self.assertEqual(bad.status_code, 403)

    def test_pages_origin_preflight_can_call_local_api(self):
        response = self.client.options(
            "/api/credentials",
            headers={
                "Origin": PAGES_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-osh-session",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), PAGES_ORIGIN)

    def test_start_route_passes_explicit_demo_confirmation(self):
        headers = {"Origin": PAGES_ORIGIN, "X-OSH-Session": SESSION}
        with patch("demo_executor.app.engine.start", return_value={"started": True}) as start:
            response = self.client.post(
                "/api/start",
                headers=headers,
                json={"confirm_demo_orders": True},
            )
        self.assertEqual(response.status_code, 200)
        start.assert_called_once_with(confirm_demo_orders=True)


if __name__ == "__main__":
    unittest.main()
