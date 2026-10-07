"""Question/SQL/Python templates for the finance and supply-chain domains.

SQL and code are correct by construction (they come from the template); every instantiated
example is still executed before it is kept.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Mapping

Slot = Callable[[sqlite3.Connection, random.Random], object]


def pick(table: str, col: str) -> Slot:
    def f(conn, rng):
        vals = [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")]
        return rng.choice(vals)
    return f


def choice(*values) -> Slot:
    return lambda conn, rng: rng.choice(values)


@dataclass(frozen=True)
class SqlTemplate:
    domain: str
    name: str
    questions: tuple[str, ...]
    sql: str
    slots: Mapping[str, Slot] = field(default_factory=dict)
    holdout: bool = False   # reserved for the unseen-template test set


@dataclass(frozen=True)
class PyTemplate:
    domain: str
    name: str
    questions: tuple[str, ...]
    tables: tuple[str, ...]          # CSV files provided to the sandbox
    code: str
    slots: Mapping[str, Slot] = field(default_factory=dict)


TOP_N = choice(3, 5, 10)
YEAR = choice(2023, 2024)


def instantiate(t, conn: sqlite3.Connection, rng: random.Random) -> tuple[str, str]:
    """(question, sql-or-code) with slots filled from the live database."""
    vals = {k: fn(conn, rng) for k, fn in t.slots.items()}
    question = rng.choice(t.questions).format(**vals)
    body = (t.sql if isinstance(t, SqlTemplate) else t.code).format(**vals)
    return question, body


# ---------------------------------------------------------------- supply chain SQL
_SC = "supply_chain"
SQL_TEMPLATES: tuple[SqlTemplate, ...] = (
    SqlTemplate(_SC, "sc_top_suppliers_po",
                ("Which {n} suppliers have the most purchase orders?", "List the top {n} suppliers by number of purchase orders."),
                "SELECT s.name, COUNT(*) AS po_count FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id GROUP BY s.name ORDER BY po_count DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate(_SC, "sc_lead_time_country",
                ("What is the average supplier lead time by country?", "Show the average lead time in days for each supplier country, longest first."),
                "SELECT country, ROUND(AVG(lead_time_days), 1) AS avg_lead_time_days FROM suppliers GROUP BY country ORDER BY avg_lead_time_days DESC"),
    SqlTemplate(_SC, "sc_on_time_rate",
                ("What percentage of received purchase orders arrived on or before the promised date, per supplier?", "Show each supplier's on-time delivery rate for received orders."),
                "SELECT s.name, ROUND(100.0 * SUM(CASE WHEN po.received_date <= po.promised_date THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_pct FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id WHERE po.status = 'received' GROUP BY s.name ORDER BY on_time_pct DESC"),
    SqlTemplate(_SC, "sc_below_reorder",
                ("Which products are below their reorder point based on total stock across all warehouses?", "List products whose total on-hand quantity is under the reorder point."),
                "SELECT p.name, SUM(i.quantity_on_hand) AS on_hand, p.reorder_point FROM inventory i JOIN products p ON p.product_id = i.product_id GROUP BY p.product_id HAVING SUM(i.quantity_on_hand) < p.reorder_point ORDER BY on_hand"),
    SqlTemplate(_SC, "sc_inventory_value_wh",
                ("What is the total inventory value in each warehouse?", "Show inventory value (quantity times unit cost) by warehouse city, highest first."),
                "SELECT w.city, ROUND(SUM(i.quantity_on_hand * p.unit_cost), 2) AS inventory_value FROM inventory i JOIN products p ON p.product_id = i.product_id JOIN warehouses w ON w.warehouse_id = i.warehouse_id GROUP BY w.city ORDER BY inventory_value DESC"),
    SqlTemplate(_SC, "sc_freight_carrier_year",
                ("What is the total freight cost per carrier in {year}?", "How much did we spend on freight with each carrier in {year}?"),
                "SELECT carrier, ROUND(SUM(freight_cost), 2) AS total_freight FROM shipments WHERE strftime('%Y', ship_date) = '{year}' GROUP BY carrier ORDER BY total_freight DESC",
                {"year": YEAR}),
    SqlTemplate(_SC, "sc_po_status",
                ("How many purchase orders are in each status?", "Count purchase orders by status."),
                "SELECT status, COUNT(*) AS orders FROM purchase_orders GROUP BY status ORDER BY orders DESC"),
    SqlTemplate(_SC, "sc_mape",
                ("What is the forecast error (MAPE) for each product, worst first? Show the top {n}.", "Which {n} products have the highest forecast MAPE?"),
                "SELECT p.name, ROUND(100.0 * AVG(ABS(f.actual_units - f.forecast_units) * 1.0 / f.actual_units), 1) AS mape_pct FROM demand_forecast f JOIN products p ON p.product_id = f.product_id WHERE f.actual_units > 0 GROUP BY p.name ORDER BY mape_pct DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate(_SC, "sc_late_count",
                ("How many purchase orders were received after the promised date?", "Count the late purchase orders."),
                "SELECT COUNT(*) AS late_orders FROM purchase_orders WHERE received_date > promised_date"),
    SqlTemplate(_SC, "sc_spend_category",
                ("What is the total purchase spend by product category?", "Show spend (quantity times unit price) per product category."),
                "SELECT p.category, ROUND(SUM(po.quantity * po.unit_price), 2) AS total_spend FROM purchase_orders po JOIN products p ON p.product_id = po.product_id GROUP BY p.category ORDER BY total_spend DESC"),
    SqlTemplate(_SC, "sc_delivery_days_carrier",
                ("What is the average delivery time in days for each carrier?", "How many days does each carrier take on average from ship date to delivery?"),
                "SELECT carrier, ROUND(AVG(julianday(delivery_date) - julianday(ship_date)), 1) AS avg_days FROM shipments GROUP BY carrier ORDER BY avg_days"),
    SqlTemplate(_SC, "sc_low_rated",
                ("Which suppliers have a rating below {r}?", "List suppliers rated under {r}, lowest first."),
                "SELECT name, country, rating FROM suppliers WHERE rating < {r} ORDER BY rating",
                {"r": choice(3.5, 4.0, 4.5)}),
    SqlTemplate(_SC, "sc_warehouse_util",
                ("What is the utilization of each warehouse (stock as a percentage of capacity)?", "Show warehouse utilization percent, highest first."),
                "SELECT w.city, ROUND(100.0 * SUM(i.quantity_on_hand) / w.capacity, 2) AS utilization_pct FROM inventory i JOIN warehouses w ON w.warehouse_id = i.warehouse_id GROUP BY w.warehouse_id ORDER BY utilization_pct DESC"),
    SqlTemplate(_SC, "sc_top_products_qty",
                ("Which {n} products have the highest total ordered quantity?", "Show the top {n} products by quantity ordered."),
                "SELECT p.name, SUM(po.quantity) AS total_qty FROM purchase_orders po JOIN products p ON p.product_id = po.product_id GROUP BY p.name ORDER BY total_qty DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate(_SC, "sc_cancelled_supplier",
                ("Which suppliers have cancelled purchase orders, and how many?", "Count cancelled orders per supplier."),
                "SELECT s.name, COUNT(*) AS cancelled FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id WHERE po.status = 'cancelled' GROUP BY s.name ORDER BY cancelled DESC"),
    SqlTemplate(_SC, "sc_monthly_orders",
                ("How many purchase orders were placed each month in {year}?", "Show the monthly count of purchase orders for {year}."),
                "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS orders FROM purchase_orders WHERE strftime('%Y', order_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}, holdout=True),
    SqlTemplate(_SC, "sc_avg_po_value_country",
                ("What is the average purchase order value by supplier country?", "Show average order value for each supplier country."),
                "SELECT s.country, ROUND(AVG(po.quantity * po.unit_price), 2) AS avg_order_value FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id GROUP BY s.country ORDER BY avg_order_value DESC",
                holdout=True),
    # ------------------------------------------------------------------ finance SQL
    SqlTemplate("finance", "fin_balance_type",
                ("What is the total balance by account type?", "Show total and average balance for each account type."),
                "SELECT account_type, ROUND(SUM(balance), 2) AS total_balance, ROUND(AVG(balance), 2) AS avg_balance FROM accounts GROUP BY account_type ORDER BY total_balance DESC"),
    SqlTemplate("finance", "fin_top_customers",
                ("Who are the top {n} customers by total account balance?", "List the {n} customers with the largest combined balances."),
                "SELECT c.name, ROUND(SUM(a.balance), 2) AS total_balance FROM accounts a JOIN customers c ON c.customer_id = a.customer_id GROUP BY c.customer_id ORDER BY total_balance DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate("finance", "fin_monthly_txn",
                ("What was the total transaction volume per month in {year}?", "Show monthly transaction totals for {year}."),
                "SELECT strftime('%Y-%m', txn_date) AS month, ROUND(SUM(amount), 2) AS volume FROM transactions WHERE strftime('%Y', txn_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}),
    SqlTemplate("finance", "fin_avg_txn_channel",
                ("What is the average transaction amount by channel?", "Compare average transaction size across channels."),
                "SELECT channel, ROUND(AVG(amount), 2) AS avg_amount, COUNT(*) AS txns FROM transactions GROUP BY channel ORDER BY avg_amount DESC"),
    SqlTemplate("finance", "fin_loans_status",
                ("How many loans are in each status and what is the total principal?", "Break down loans by status with count and principal."),
                "SELECT status, COUNT(*) AS loans, ROUND(SUM(principal), 2) AS total_principal FROM loans GROUP BY status ORDER BY total_principal DESC"),
    SqlTemplate("finance", "fin_rate_status",
                ("What is the average interest rate by loan status?", "Show the mean interest rate for each loan status."),
                "SELECT status, ROUND(AVG(interest_rate), 2) AS avg_rate FROM loans GROUP BY status ORDER BY avg_rate DESC"),
    SqlTemplate("finance", "fin_overdue",
                ("Which {n} loans have the most late repayments?", "Show the {n} loans with the highest number of repayments paid after the due date."),
                "SELECT loan_id, COUNT(*) AS late_payments FROM repayments WHERE paid_date > due_date GROUP BY loan_id ORDER BY late_payments DESC, loan_id LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate("finance", "fin_principal_segment",
                ("What is the total loan principal by customer segment?", "Show loan principal per customer segment, largest first."),
                "SELECT c.segment, ROUND(SUM(l.principal), 2) AS total_principal FROM loans l JOIN customers c ON c.customer_id = l.customer_id GROUP BY c.segment ORDER BY total_principal DESC"),
    SqlTemplate("finance", "fin_high_risk",
                ("Which customers have a {risk} risk rating and what is their total balance?", "List {risk}-risk customers with their combined account balances."),
                "SELECT c.name, ROUND(SUM(a.balance), 2) AS total_balance FROM customers c JOIN accounts a ON a.customer_id = c.customer_id WHERE c.risk_rating = '{risk}' GROUP BY c.customer_id ORDER BY total_balance DESC",
                {"risk": choice("High", "Medium")}),
    SqlTemplate("finance", "fin_net_cash_flow",
                ("What is the net cash flow (credits minus debits) for the top {n} accounts?", "Show net cash flow per account, highest {n} first."),
                "SELECT account_id, ROUND(SUM(CASE WHEN txn_type = 'credit' THEN amount ELSE -amount END), 2) AS net_cash_flow FROM transactions GROUP BY account_id ORDER BY net_cash_flow DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate("finance", "fin_largest_txn",
                ("What are the {n} largest transactions?", "List the {n} biggest transactions with their channel."),
                "SELECT txn_id, txn_date, amount, channel FROM transactions ORDER BY amount DESC LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate("finance", "fin_repayment_ratio",
                ("What share of the amount due has been paid, per loan? Show the {n} lowest.", "Which {n} loans have the lowest repayment ratio?"),
                "SELECT loan_id, ROUND(100.0 * SUM(amount_paid) / SUM(amount_due), 1) AS paid_pct FROM repayments GROUP BY loan_id ORDER BY paid_pct, loan_id LIMIT {n}",
                {"n": TOP_N}),
    SqlTemplate("finance", "fin_gl_totals",
                ("What are the total debits and credits by account code?", "Summarize the general ledger by account code."),
                "SELECT account_code, ROUND(SUM(debit), 2) AS total_debit, ROUND(SUM(credit), 2) AS total_credit FROM gl_entries GROUP BY account_code ORDER BY account_code"),
    SqlTemplate("finance", "fin_merchant_total",
                ("How much was spent in each merchant category?", "Show total transaction amount by merchant category."),
                "SELECT merchant_category, ROUND(SUM(amount), 2) AS total_spent FROM transactions GROUP BY merchant_category ORDER BY total_spent DESC"),
    SqlTemplate("finance", "fin_late_rate_segment",
                ("What percentage of repayments were paid late, by customer segment?", "Show the late-payment rate for each customer segment."),
                "SELECT c.segment, ROUND(100.0 * SUM(CASE WHEN r.paid_date > r.due_date THEN 1 ELSE 0 END) / COUNT(*), 1) AS late_pct FROM repayments r JOIN loans l ON l.loan_id = r.loan_id JOIN customers c ON c.customer_id = l.customer_id WHERE r.paid_date IS NOT NULL GROUP BY c.segment ORDER BY late_pct DESC",
                holdout=True),
    SqlTemplate("finance", "fin_balance_country",
                ("What is the total account balance by customer country?", "Show the combined balance for each country, largest first."),
                "SELECT c.country, ROUND(SUM(a.balance), 2) AS total_balance FROM accounts a JOIN customers c ON c.customer_id = a.customer_id GROUP BY c.country ORDER BY total_balance DESC",
                holdout=True),
)

EXTRA_SQL_TEMPLATES: tuple[SqlTemplate, ...] = (
    SqlTemplate(_SC, "sc_monthly_receipts",
                ("How many purchase orders were received in each month of {year}?", "Show the number of orders received per month in {year}."),
                "SELECT strftime('%Y-%m', received_date) AS month, COUNT(*) AS received_orders FROM purchase_orders WHERE received_date IS NOT NULL AND strftime('%Y', received_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}),
    SqlTemplate(_SC, "sc_monthly_shipments",
                ("How many shipments left each month in {year}?", "Count shipments by ship month for {year}."),
                "SELECT strftime('%Y-%m', ship_date) AS month, COUNT(*) AS shipments FROM shipments WHERE strftime('%Y', ship_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}),
    SqlTemplate(_SC, "sc_monthly_freight",
                ("What was the total freight cost per month in {year}?", "Show freight spend by month for {year}."),
                "SELECT strftime('%Y-%m', ship_date) AS month, ROUND(SUM(freight_cost), 2) AS freight FROM shipments WHERE strftime('%Y', ship_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}),
    SqlTemplate(_SC, "sc_suppliers_many_pos",
                ("Which suppliers have more than {k} purchase orders?", "List suppliers with over {k} orders and their order counts."),
                "SELECT s.name, COUNT(*) AS orders FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id GROUP BY s.supplier_id HAVING COUNT(*) > {k} ORDER BY orders DESC",
                {"k": choice(3, 5)}),
    SqlTemplate(_SC, "sc_products_multi_wh",
                ("Which products are stocked in more than {k} warehouses?", "List products held in over {k} warehouses."),
                "SELECT p.name, COUNT(DISTINCT i.warehouse_id) AS warehouses FROM inventory i JOIN products p ON p.product_id = i.product_id WHERE i.quantity_on_hand > 0 GROUP BY p.product_id HAVING COUNT(DISTINCT i.warehouse_id) > {k} ORDER BY warehouses DESC, p.name",
                {"k": choice(1, 2)}),
    SqlTemplate(_SC, "sc_freight_supplier_country",
                ("What is the average freight cost per shipment by supplier country?", "Show mean freight cost for each supplier country."),
                "SELECT s.country, ROUND(AVG(sh.freight_cost), 2) AS avg_freight FROM shipments sh JOIN purchase_orders po ON po.po_id = sh.po_id JOIN suppliers s ON s.supplier_id = po.supplier_id GROUP BY s.country ORDER BY avg_freight DESC"),
    SqlTemplate(_SC, "sc_order_size_bands",
                ("How many purchase orders are small (under 100 units), medium (100 to 299) or large (300 or more)?", "Group orders into small, medium and large by quantity and count them."),
                "SELECT CASE WHEN quantity >= 300 THEN 'large' WHEN quantity >= 100 THEN 'medium' ELSE 'small' END AS size, COUNT(*) AS orders FROM purchase_orders GROUP BY size ORDER BY orders DESC"),
    SqlTemplate(_SC, "sc_spend_supplier_country",
                ("What is the total purchase spend by supplier country?", "Show spend per supplier country, largest first."),
                "SELECT s.country, ROUND(SUM(po.quantity * po.unit_price), 2) AS total_spend FROM purchase_orders po JOIN suppliers s ON s.supplier_id = po.supplier_id GROUP BY s.country ORDER BY total_spend DESC"),
    SqlTemplate("finance", "fin_monthly_debit_credit",
                ("Show monthly debit and credit totals for {year}.", "What were the debit and credit amounts by month in {year}?"),
                "SELECT strftime('%Y-%m', txn_date) AS month, ROUND(SUM(CASE WHEN txn_type = 'debit' THEN amount ELSE 0 END), 2) AS debits, ROUND(SUM(CASE WHEN txn_type = 'credit' THEN amount ELSE 0 END), 2) AS credits FROM transactions WHERE strftime('%Y', txn_date) = '{year}' GROUP BY month ORDER BY month",
                {"year": YEAR}),
    SqlTemplate("finance", "fin_loans_many_late",
                ("Which loans have more than {k} late repayments?", "List loans with over {k} payments made after the due date."),
                "SELECT loan_id, COUNT(*) AS late_payments FROM repayments WHERE paid_date > due_date GROUP BY loan_id HAVING COUNT(*) > {k} ORDER BY late_payments DESC, loan_id",
                {"k": choice(1, 2)}),
    SqlTemplate("finance", "fin_customers_multi_accounts",
                ("Which customers have more than one account?", "List customers holding multiple accounts and how many."),
                "SELECT c.name, COUNT(*) AS accounts FROM accounts a JOIN customers c ON c.customer_id = a.customer_id GROUP BY c.customer_id HAVING COUNT(*) > 1 ORDER BY accounts DESC, c.name"),
    SqlTemplate("finance", "fin_rate_country",
                ("What is the average loan interest rate by customer country?", "Show the mean interest rate of loans for each customer country."),
                "SELECT c.country, ROUND(AVG(l.interest_rate), 2) AS avg_rate FROM loans l JOIN customers c ON c.customer_id = l.customer_id GROUP BY c.country ORDER BY avg_rate DESC"),
    SqlTemplate("finance", "fin_balance_bands",
                ("How many accounts have a high (over 50000), medium (10000 to 50000) or low balance?", "Group accounts into balance bands and count them."),
                "SELECT CASE WHEN balance > 50000 THEN 'high' WHEN balance >= 10000 THEN 'medium' ELSE 'low' END AS band, COUNT(*) AS accounts FROM accounts GROUP BY band ORDER BY accounts DESC"),
    SqlTemplate("finance", "fin_gl_net_cost_center",
                ("What is the net amount (debits minus credits) by cost center?", "Show net ledger balance per cost center."),
                "SELECT cost_center, ROUND(SUM(debit) - SUM(credit), 2) AS net_amount FROM gl_entries GROUP BY cost_center ORDER BY net_amount DESC"),
    SqlTemplate("finance", "fin_old_accounts",
                ("How many accounts of each type were opened before {y}?", "Count accounts opened before {y} by account type."),
                "SELECT account_type, COUNT(*) AS accounts FROM accounts WHERE opened_date < '{y}-01-01' GROUP BY account_type ORDER BY accounts DESC",
                {"y": choice(2020, 2021, 2022)}),
)
SQL_TEMPLATES = SQL_TEMPLATES + EXTRA_SQL_TEMPLATES

# ------------------------------------------------------------------- pandas tasks
_PIPE = "print(res.to_csv(index=False, sep='|'))"
PY_TEMPLATES: tuple[PyTemplate, ...] = (
    PyTemplate(_SC, "py_spend_supplier",
               ("Calculate the total spend (quantity x unit_price) for each supplier_id and show the top {n}.",),
               ("purchase_orders",),
               "import pandas as pd\npo = pd.read_csv('purchase_orders.csv')\npo['spend'] = po['quantity'] * po['unit_price']\n"
               "res = po.groupby('supplier_id', as_index=False)['spend'].sum().round(2).sort_values('spend', ascending=False).head({n})\n" + _PIPE,
               {"n": TOP_N}),
    PyTemplate(_SC, "py_late_rate",
               ("For received orders, what percentage were late for each supplier_id? Show the {n} worst.",),
               ("purchase_orders",),
               "import pandas as pd\npo = pd.read_csv('purchase_orders.csv', parse_dates=['promised_date', 'received_date'])\n"
               "rec = po[po['status'] == 'received'].copy()\nrec['late'] = rec['received_date'] > rec['promised_date']\n"
               "res = rec.groupby('supplier_id', as_index=False)['late'].mean()\nres['late_pct'] = (res['late'] * 100).round(1)\n"
               "res = res[['supplier_id', 'late_pct']].sort_values('late_pct', ascending=False).head({n})\n" + _PIPE,
               {"n": TOP_N}),
    PyTemplate(_SC, "py_avg_cost_category",
               ("What is the average unit cost per product category?",),
               ("products",),
               "import pandas as pd\nproducts = pd.read_csv('products.csv')\n"
               "res = products.groupby('category', as_index=False)['unit_cost'].mean().round(2).sort_values('unit_cost', ascending=False)\n" + _PIPE),
    PyTemplate(_SC, "py_forecast_bias",
               ("Compute the average forecast bias (actual minus forecast) for each product_id and show the {n} most under-forecast.",),
               ("demand_forecast",),
               "import pandas as pd\nf = pd.read_csv('demand_forecast.csv')\nf['bias'] = f['actual_units'] - f['forecast_units']\n"
               "res = f.groupby('product_id', as_index=False)['bias'].mean().round(1).sort_values('bias', ascending=False).head({n})\n" + _PIPE,
               {"n": TOP_N}),
    PyTemplate(_SC, "py_monthly_orders",
               ("How many purchase orders were placed in each month of {year}?",),
               ("purchase_orders",),
               "import pandas as pd\npo = pd.read_csv('purchase_orders.csv', parse_dates=['order_date'])\npo = po[po['order_date'].dt.year == {year}]\n"
               "res = po.groupby(po['order_date'].dt.strftime('%Y-%m')).size().reset_index(name='orders')\nres.columns = ['month', 'orders']\n" + _PIPE,
               {"year": YEAR}),
    PyTemplate(_SC, "py_stock_warehouse",
               ("What is the total quantity on hand in each warehouse?",),
               ("inventory",),
               "import pandas as pd\ninv = pd.read_csv('inventory.csv')\nres = inv.groupby('warehouse_id', as_index=False)['quantity_on_hand'].sum().sort_values('quantity_on_hand', ascending=False)\n" + _PIPE),
    PyTemplate(_SC, "py_qty_outliers",
               ("How many purchase orders have a quantity more than 2 standard deviations above the mean?",),
               ("purchase_orders",),
               "import pandas as pd\npo = pd.read_csv('purchase_orders.csv')\nz = (po['quantity'] - po['quantity'].mean()) / po['quantity'].std()\n"
               "res = pd.DataFrame({{'outlier_orders': [int((z > 2).sum())]}})\n" + _PIPE),
    PyTemplate(_SC, "py_price_qty_corr",
               ("What is the correlation between order quantity and unit price?",),
               ("purchase_orders",),
               "import pandas as pd\npo = pd.read_csv('purchase_orders.csv')\nres = pd.DataFrame({{'correlation': [round(po['quantity'].corr(po['unit_price']), 3)]}})\n" + _PIPE),
    PyTemplate("finance", "py_balance_type",
               ("What is the average balance for each account type?",),
               ("accounts",),
               "import pandas as pd\nacc = pd.read_csv('accounts.csv')\nres = acc.groupby('account_type', as_index=False)['balance'].mean().round(2).sort_values('balance', ascending=False)\n" + _PIPE),
    PyTemplate("finance", "py_monthly_credits",
               ("What is the total credit transaction amount per month in {year}?",),
               ("transactions",),
               "import pandas as pd\ntx = pd.read_csv('transactions.csv', parse_dates=['txn_date'])\ntx = tx[(tx['txn_type'] == 'credit') & (tx['txn_date'].dt.year == {year})]\n"
               "res = tx.groupby(tx['txn_date'].dt.strftime('%Y-%m'))['amount'].sum().round(2).reset_index()\nres.columns = ['month', 'credit_total']\n" + _PIPE,
               {"year": YEAR}),
    PyTemplate("finance", "py_top_customers",
               ("Which {n} customers hold the largest total balance? Join accounts with customers.",),
               ("accounts", "customers"),
               "import pandas as pd\nacc = pd.read_csv('accounts.csv')\ncust = pd.read_csv('customers.csv')\nm = acc.merge(cust, on='customer_id')\n"
               "res = m.groupby('name', as_index=False)['balance'].sum().round(2).sort_values('balance', ascending=False).head({n})\n" + _PIPE,
               {"n": TOP_N}),
    PyTemplate("finance", "py_weighted_rate",
               ("What is the principal-weighted average interest rate across all loans?",),
               ("loans",),
               "import pandas as pd\nloans = pd.read_csv('loans.csv')\nrate = (loans['interest_rate'] * loans['principal']).sum() / loans['principal'].sum()\n"
               "res = pd.DataFrame({{'weighted_avg_rate': [round(rate, 2)]}})\n" + _PIPE),
    PyTemplate("finance", "py_late_payments",
               ("Count the late repayments (paid after the due date) for each loan_id and show the top {n}.",),
               ("repayments",),
               "import pandas as pd\nr = pd.read_csv('repayments.csv', parse_dates=['due_date', 'paid_date'])\nlate = r[r['paid_date'] > r['due_date']]\n"
               "res = late.groupby('loan_id').size().reset_index(name='late_payments').sort_values(['late_payments', 'loan_id'], ascending=[False, True]).head({n})\n" + _PIPE,
               {"n": TOP_N}),
    PyTemplate("finance", "py_channel_share",
               ("What percentage of the total transaction amount goes through each channel?",),
               ("transactions",),
               "import pandas as pd\ntx = pd.read_csv('transactions.csv')\nshare = tx.groupby('channel')['amount'].sum() / tx['amount'].sum() * 100\n"
               "res = share.round(1).reset_index()\nres.columns = ['channel', 'share_pct']\nres = res.sort_values('share_pct', ascending=False)\n" + _PIPE),
    PyTemplate("finance", "py_large_txn",
               ("How many transactions above {t} are there in each merchant category?",),
               ("transactions",),
               "import pandas as pd\ntx = pd.read_csv('transactions.csv')\nbig = tx[tx['amount'] > {t}]\n"
               "res = big.groupby('merchant_category').size().reset_index(name='large_txns').sort_values(['large_txns', 'merchant_category'], ascending=[False, True])\n" + _PIPE,
               {"t": choice(1000, 2000, 3000)}),
)
