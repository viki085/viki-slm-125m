"""Synthetic finance and supply-chain databases (SQLite) for programmatic SFT data."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from typing import Callable

SUPPLY_CHAIN_DDL = """CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, name TEXT, country TEXT, rating REAL, lead_time_days INTEGER);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_cost REAL, reorder_point INTEGER);
CREATE TABLE warehouses (warehouse_id INTEGER PRIMARY KEY, city TEXT, capacity INTEGER);
CREATE TABLE inventory (product_id INTEGER, warehouse_id INTEGER, quantity_on_hand INTEGER, last_counted TEXT);
CREATE TABLE purchase_orders (po_id INTEGER PRIMARY KEY, supplier_id INTEGER, product_id INTEGER, order_date TEXT, quantity INTEGER, unit_price REAL, status TEXT, promised_date TEXT, received_date TEXT);
CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, po_id INTEGER, carrier TEXT, ship_date TEXT, delivery_date TEXT, freight_cost REAL);
CREATE TABLE demand_forecast (product_id INTEGER, month TEXT, forecast_units INTEGER, actual_units INTEGER);"""

FINANCE_DDL = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, segment TEXT, country TEXT, risk_rating TEXT);
CREATE TABLE accounts (account_id INTEGER PRIMARY KEY, customer_id INTEGER, account_type TEXT, balance REAL, opened_date TEXT);
CREATE TABLE transactions (txn_id INTEGER PRIMARY KEY, account_id INTEGER, txn_date TEXT, amount REAL, txn_type TEXT, channel TEXT, merchant_category TEXT);
CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, customer_id INTEGER, principal REAL, interest_rate REAL, term_months INTEGER, status TEXT, origination_date TEXT);
CREATE TABLE repayments (repayment_id INTEGER PRIMARY KEY, loan_id INTEGER, due_date TEXT, paid_date TEXT, amount_due REAL, amount_paid REAL);
CREATE TABLE gl_entries (entry_id INTEGER PRIMARY KEY, account_code TEXT, entry_date TEXT, debit REAL, credit REAL, cost_center TEXT);"""

COUNTRIES = ["Germany", "China", "India", "Mexico", "Vietnam", "Poland", "Brazil", "Turkey", "USA", "Japan"]
CATEGORIES = ["Electronics", "Packaging", "Raw Materials", "Apparel", "Machinery", "Chemicals"]
CITIES = ["Rotterdam", "Chicago", "Singapore", "Dallas", "Hamburg", "Shenzhen", "Atlanta"]
CARRIERS = ["DHL", "FedEx", "Maersk", "UPS", "DB Schenker"]
PO_STATUSES = ["received", "received", "received", "open", "cancelled"]
SEGMENTS = ["Retail", "SME", "Corporate", "Private"]
RISKS = ["Low", "Low", "Medium", "High"]
ACCOUNT_TYPES = ["Checking", "Savings", "Credit", "Investment"]
CHANNELS = ["Online", "Branch", "ATM", "Mobile"]
MERCHANTS = ["Groceries", "Travel", "Utilities", "Dining", "Retail", "Healthcare"]
LOAN_STATUSES = ["current", "current", "late", "default", "paid off"]
COST_CENTERS = ["Operations", "Sales", "IT", "Finance", "Logistics"]
ACCOUNT_CODES = ["1000", "2000", "4000", "5000", "6100", "6200"]
WORDS = ["Apex", "Nova", "Orion", "Vertex", "Summit", "Atlas", "Zenith", "Delta", "Pioneer", "Prime",
         "Harbor", "Crest", "Titan", "Vector", "Quantum", "Anchor"]
SURNAMES = ["Smith", "Garcia", "Chen", "Patel", "Kowalski", "Silva", "Nguyen", "Mueller", "Okafor", "Tanaka"]


def _day(rng: random.Random, start: date = date(2023, 1, 1), span: int = 700) -> date:
    return start + timedelta(days=rng.randrange(span))


def _populate(conn: sqlite3.Connection, table: str, rows: list[tuple]) -> None:
    marks = ",".join("?" * len(rows[0]))
    conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)


def make_supply_chain_db(rng: random.Random) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SUPPLY_CHAIN_DDL)
    n_sup, n_prod, n_wh = rng.randint(8, 14), rng.randint(10, 18), rng.randint(3, 5)
    _populate(conn, "suppliers", [(i, f"{rng.choice(WORDS)} {rng.choice(['Supply', 'Logistics', 'Components', 'Industries'])} {i}",
                                   rng.choice(COUNTRIES), round(rng.uniform(2.5, 5.0), 1), rng.randint(3, 45))
                                  for i in range(1, n_sup + 1)])
    _populate(conn, "products", [(i, f"{rng.choice(WORDS)} {rng.choice(['Widget', 'Module', 'Panel', 'Kit', 'Pack'])} {i}",
                                  rng.choice(CATEGORIES), round(rng.uniform(1.5, 120.0), 2), rng.randint(20, 200))
                                 for i in range(1, n_prod + 1)])
    _populate(conn, "warehouses", [(i, CITIES[i - 1], rng.randint(5_000, 40_000)) for i in range(1, n_wh + 1)])
    _populate(conn, "inventory", [(p, w, rng.randint(0, 60), _day(rng).isoformat())
                                  for p in range(1, n_prod + 1) for w in range(1, n_wh + 1)])
    pos = []
    for po in range(1, rng.randint(60, 110) + 1):
        order = _day(rng)
        promised = order + timedelta(days=rng.randint(5, 40))
        status = rng.choice(PO_STATUSES)
        received = (promised + timedelta(days=rng.randint(-4, 12))).isoformat() if status == "received" else None
        pos.append((po, rng.randint(1, n_sup), rng.randint(1, n_prod), order.isoformat(), rng.randint(10, 500),
                    round(rng.uniform(1.5, 130.0), 2), status, promised.isoformat(), received))
    _populate(conn, "purchase_orders", pos)
    ships = []
    for sid, po in enumerate([p for p in pos if p[6] == "received"], start=1):
        ship = date.fromisoformat(po[3]) + timedelta(days=rng.randint(1, 10))
        ships.append((sid, po[0], rng.choice(CARRIERS), ship.isoformat(),
                      (ship + timedelta(days=rng.randint(2, 20))).isoformat(), round(rng.uniform(40, 2500), 2)))
    _populate(conn, "shipments", ships)
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    _populate(conn, "demand_forecast", [(p, m, (f := rng.randint(50, 600)), max(1, int(f * rng.uniform(0.7, 1.3))))
                                        for p in range(1, n_prod + 1) for m in months])
    conn.commit()
    return conn


def make_finance_db(rng: random.Random) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(FINANCE_DDL)
    n_cust = rng.randint(15, 30)
    _populate(conn, "customers", [(i, f"{rng.choice(['Alex', 'Sam', 'Maria', 'Wei', 'Priya', 'Omar', 'Lena'])} {rng.choice(SURNAMES)}",
                                   rng.choice(SEGMENTS), rng.choice(COUNTRIES), rng.choice(RISKS))
                                  for i in range(1, n_cust + 1)])
    accts = [(a, rng.randint(1, n_cust), rng.choice(ACCOUNT_TYPES), round(rng.uniform(100, 90_000), 2),
              _day(rng, date(2018, 1, 1), 1800).isoformat()) for a in range(1, n_cust * 2 + 1)]
    _populate(conn, "accounts", accts)
    _populate(conn, "transactions", [(t, rng.randint(1, len(accts)), _day(rng).isoformat(),
                                      round(rng.uniform(5, 5_000), 2), rng.choice(["debit", "credit"]),
                                      rng.choice(CHANNELS), rng.choice(MERCHANTS))
                                     for t in range(1, rng.randint(150, 300) + 1)])
    loans = [(l, rng.randint(1, n_cust), round(rng.uniform(2_000, 250_000), 2), round(rng.uniform(2.5, 14.0), 2),
              rng.choice([12, 24, 36, 60, 120]), rng.choice(LOAN_STATUSES), _day(rng, date(2021, 1, 1), 1000).isoformat())
             for l in range(1, rng.randint(15, 30) + 1)]
    _populate(conn, "loans", loans)
    reps = []
    for r in range(1, rng.randint(80, 140) + 1):
        due = _day(rng, date(2023, 1, 1), 600)
        due_amt = round(rng.uniform(150, 3_000), 2)
        paid = None if rng.random() < 0.1 else (due + timedelta(days=rng.randint(-3, 25))).isoformat()
        reps.append((r, rng.randint(1, len(loans)), due.isoformat(), paid, due_amt,
                     0.0 if paid is None else round(due_amt * rng.choice([1, 1, 1, 0.5, 0.9]), 2)))
    _populate(conn, "repayments", reps)
    gl = []
    for e in range(1, rng.randint(80, 150) + 1):
        amount = round(rng.uniform(0, 20_000), 2)
        debit, credit = (amount, 0.0) if rng.random() < 0.5 else (0.0, amount)
        gl.append((e, rng.choice(ACCOUNT_CODES), _day(rng).isoformat(), debit, credit, rng.choice(COST_CENTERS)))
    _populate(conn, "gl_entries", gl)
    conn.commit()
    return conn


DOMAINS: dict[str, tuple[str, Callable[[random.Random], sqlite3.Connection]]] = {
    "supply_chain": (SUPPLY_CHAIN_DDL, make_supply_chain_db),
    "finance": (FINANCE_DDL, make_finance_db),
}


def table_csv(conn: sqlite3.Connection, table: str) -> str:
    """The table as CSV text (header + rows) for sandboxed pandas tasks."""
    cur = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    lines = [",".join(cols)]
    for row in cur.fetchall():
        lines.append(",".join("" if v is None else str(v) for v in row))
    return "\n".join(lines) + "\n"
