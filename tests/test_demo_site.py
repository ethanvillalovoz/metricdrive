from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metricdrive.demo_site import generate_demo_site
from metricdrive.samples import synthetic_scenarios


class DemoSiteTests(unittest.TestCase):
    def test_generate_demo_site_writes_static_explorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "demo"

            payload = generate_demo_site(synthetic_scenarios(), output, epochs=5)

            self.assertEqual(payload["format"], "metricdrive.demo.v1")
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "styles.css").exists())
            self.assertTrue((output / "app.js").exists())
            self.assertTrue((output / "scenarios.json").exists())
            self.assertTrue((output / "assets" / "metricdrive-explorer.svg").exists())
            self.assertIn(
                "MetricDrive Explorer",
                (output / "index.html").read_text(encoding="utf-8"),
            )
            self.assertIn(
                'rel="icon"',
                (output / "index.html").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "ethanvillalovoz.com/metricdrive",
                (output / "README.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
