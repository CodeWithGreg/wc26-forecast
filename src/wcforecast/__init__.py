"""wcforecast — probabilistic score forecasting for the FIFA World Cup 2026.

A forecasting package built around two complementary scoreline models:

* a time-decayed Dixon-Coles bivariate Poisson model (interpretable team
  abilities, analytic posterior for epistemic uncertainty), and
* a gradient-boosted Poisson model (LightGBM) on engineered, leak-free
  features (Elo, form, match context), bagged for epistemic uncertainty,

combined through a log-linear ensemble pool. Forecasts are full scoreline
distributions, from which 1X2 probabilities, goal intervals, Monte-Carlo
tournament simulations and Mon-Petit-Prono optimal picks are derived.

Author: gclaus <https://www.linkedin.com/in/gregoireclaus/>
"""

__version__ = "0.1.0"
__author__ = "gclaus"

from wcforecast.config import Settings, get_settings  # noqa: F401
