from qunimbus.synthetic.order_book import OrderBook


def test_order_matching_is_deterministic():
    book = OrderBook()
    book.submit("a", "buy", 10.0, 5)
    book.submit("b", "sell", 9.0, 3)
    book.submit("c", "sell", 9.5, 3)

    trades = book.match()

    assert len(trades) == 1
    assert trades[0].quantity == 3
    assert trades[0].price == 9.5
    depth = book.depth()
    assert depth["bid_qty"] == 2
    assert depth["ask_qty"] == 3
