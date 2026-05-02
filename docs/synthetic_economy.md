# Synthetic Economy

The synthetic economy module (`qunimbus.synthetic`) models deterministic market microstructure and credit links for campaign and scenario use.

- **OrderBook**: price-time priority matching, generating reproducible trades and depth metrics.
- **MarketVenue**: wraps the book with mid-price and trade capture.
- **EconomicAgent**: capital + position accounting with deterministic mark-to-market.
- **CreditNetwork**: exposures and contagion calculations on default events.
- **Shocks**: price and liquidity shocks applied in a controlled manner.

These components can be calibrated using normalized QReal feeds while remaining fully deterministic for replay.
