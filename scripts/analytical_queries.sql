-- =====================================================================
-- AML, Fraud & Mule Account Detection Analytics - analytical SQL (SQLite)
-- Database: data/processed/banking_risk.db (built by `python run_pipeline.py`)
--
-- Tables
--   accounts(account_id, is_suspicious, init_balance, fraud_step_raw, first_step, last_step)
--   transactions(tx_id, src, dst, amount, step, week, is_self_transfer, dup_rank, is_repeat_copy)
--   account_week_scores(account_id, week, run_step, period, group, segment, risk_score, alert, sig_*, contrib_*)
--   alert_queue(alert_id, week, account_id, risk_level, risk_score, evidence_strength, ...)
--   evaluation_labels(tx_id, src_suspicious, dst_suspicious, proxy_typology_tx)   -- evaluation only
--
-- Conventions: `step` is an integer simulation day; week = (step-1)/7 + 1.
-- Queries marked [EVALUATION] join ground-truth labels and must never feed detection.
-- Each query starts with a "-- name:" line so sql_runner.py can execute and export it.
-- =====================================================================

-- name: q01_data_overview
-- Q: How big is the dataset and what does it cover?
SELECT COUNT(*)                                   AS transactions,
       COUNT(DISTINCT src)                        AS sending_accounts,
       COUNT(DISTINCT dst)                        AS receiving_accounts,
       ROUND(SUM(amount), 2)                      AS total_value,
       ROUND(AVG(amount), 2)                      AS mean_amount,
       MIN(step)                                  AS first_step,
       MAX(step)                                  AS last_step,
       SUM(is_self_transfer)                      AS self_transfers,
       SUM(is_repeat_copy)                        AS exact_duplicate_copies,
       (SELECT COUNT(*) FROM accounts)            AS accounts,
       (SELECT SUM(is_suspicious) FROM accounts)  AS labelled_accounts
FROM transactions;

-- name: q02_weekly_volume_trend
-- Q: How does weekly volume evolve, and how fast does it change week over week?
WITH weekly AS (
    SELECT week, COUNT(*) AS tx, ROUND(SUM(amount), 2) AS value, COUNT(DISTINCT src) AS active_senders
    FROM transactions
    GROUP BY week
)
SELECT week, tx, value, active_senders,
       LAG(tx) OVER (ORDER BY week)                                           AS prev_week_tx,
       ROUND(100.0 * (tx - LAG(tx) OVER (ORDER BY week)) / LAG(tx) OVER (ORDER BY week), 1) AS wow_change_pct,
       SUM(tx) OVER (ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_tx,
       ROUND(AVG(tx) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 1) AS rolling_4wk_avg
FROM weekly
ORDER BY week;

-- name: q03_highest_velocity_accounts
-- Q: Which accounts have the highest weekly transaction velocity (either direction)?
WITH legs AS (
    SELECT src AS account_id, week FROM transactions WHERE is_self_transfer = 0
    UNION ALL
    SELECT dst AS account_id, week FROM transactions WHERE is_self_transfer = 0
),
weekly AS (
    SELECT account_id, week, COUNT(*) AS tx_in_week FROM legs GROUP BY account_id, week
),
peak AS (
    SELECT account_id, MAX(tx_in_week) AS peak_week_tx, SUM(tx_in_week) AS total_tx,
           COUNT(*) AS active_weeks
    FROM weekly GROUP BY account_id
)
SELECT account_id, peak_week_tx, total_tx, active_weeks,
       ROUND(1.0 * total_tx / active_weeks, 2) AS tx_per_active_week,
       RANK() OVER (ORDER BY peak_week_tx DESC) AS velocity_rank
FROM peak
ORDER BY velocity_rank
LIMIT 25;

-- name: q04_inflow_outflow_behaviour
-- Q: Which accounts move money through (in ~ out) versus accumulate or distribute?
WITH flows AS (
    SELECT account_id, SUM(inflow) AS inflow, SUM(outflow) AS outflow,
           SUM(n_in) AS n_in, SUM(n_out) AS n_out
    FROM (
        SELECT dst AS account_id, amount AS inflow, 0 AS outflow, 1 AS n_in, 0 AS n_out
        FROM transactions WHERE is_self_transfer = 0
        UNION ALL
        SELECT src, 0, amount, 0, 1 FROM transactions WHERE is_self_transfer = 0
    ) GROUP BY account_id
),
classified AS (
    SELECT account_id, ROUND(inflow, 2) AS inflow, ROUND(outflow, 2) AS outflow, n_in, n_out,
           ROUND(MIN(inflow, outflow) / MAX(inflow, outflow), 3) AS balance_ratio,
           CASE
               WHEN inflow = 0 OR outflow = 0              THEN 'one-directional'
               WHEN MIN(inflow, outflow) / MAX(inflow, outflow) >= 0.9 THEN 'pass-through (in ~ out)'
               WHEN inflow > outflow                       THEN 'net accumulator'
               ELSE 'net distributor'
           END AS flow_profile
    FROM flows
)
SELECT flow_profile, COUNT(*) AS accounts, ROUND(AVG(n_in + n_out), 1) AS avg_transactions,
       ROUND(AVG(inflow), 1) AS avg_inflow, ROUND(AVG(outflow), 1) AS avg_outflow
FROM classified
GROUP BY flow_profile
ORDER BY accounts DESC;

-- name: q05_counterparty_concentration
-- Q: For busy senders, how dependent are outflows on a single receiver?
WITH pair AS (
    SELECT src, dst, SUM(amount) AS value, COUNT(*) AS n
    FROM transactions WHERE is_self_transfer = 0
    GROUP BY src, dst
),
ranked AS (
    SELECT src, dst, value, n,
           ROW_NUMBER() OVER (PARTITION BY src ORDER BY value DESC) AS rn,
           SUM(value) OVER (PARTITION BY src)                      AS total_out,
           COUNT(*)  OVER (PARTITION BY src)                       AS receivers
    FROM pair
)
SELECT src AS account_id, receivers, ROUND(total_out, 2) AS total_out, dst AS top_receiver,
       ROUND(100.0 * value / total_out, 1) AS top_receiver_share_pct, n AS transfers_to_top
FROM ranked
WHERE rn = 1 AND receivers >= 10
ORDER BY top_receiver_share_pct DESC
LIMIT 25;

-- name: q06_sudden_activity_change
-- Q: Which account-weeks are far above the account's own TRAILING (past-only) average?
-- The window uses 4 PRECEDING .. 1 PRECEDING, so the current week is never in its own baseline.
WITH legs AS (
    SELECT src AS account_id, week FROM transactions WHERE is_self_transfer = 0
    UNION ALL
    SELECT dst, week FROM transactions WHERE is_self_transfer = 0
),
weekly AS (
    SELECT account_id, week, COUNT(*) AS tx FROM legs GROUP BY account_id, week
),
baseline AS (
    SELECT account_id, week, tx,
           AVG(tx)   OVER (PARTITION BY account_id ORDER BY week ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS trailing_avg,
           COUNT(tx) OVER (PARTITION BY account_id ORDER BY week ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS weeks_in_baseline
    FROM weekly
)
SELECT account_id, week, tx, ROUND(trailing_avg, 2) AS trailing_avg,
       ROUND(tx / trailing_avg, 2) AS spike_ratio
FROM baseline
WHERE weeks_in_baseline >= 2 AND tx >= 5 AND tx / trailing_avg >= 3
ORDER BY spike_ratio DESC, tx DESC
LIMIT 25;

-- name: q07_highest_value_relationships
-- Q: Which sender-receiver relationships carry the most value, and how concentrated is value?
WITH pair AS (
    SELECT src, dst, COUNT(*) AS n, SUM(amount) AS value, MIN(step) AS first_step, MAX(step) AS last_step
    FROM transactions WHERE is_self_transfer = 0 GROUP BY src, dst
),
tot AS (SELECT SUM(value) AS total FROM pair)
SELECT src, dst, n, ROUND(value, 2) AS value, first_step, last_step,
       ROUND(100.0 * value / total, 4) AS pct_of_all_value,
       ROUND(100.0 * SUM(value) OVER (ORDER BY value DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / total, 3)
           AS cumulative_pct
FROM pair, tot
ORDER BY value DESC
LIMIT 25;

-- name: q08_rapid_forwarding
-- Q: Which accounts send money out within 3 days of receiving it (holding-time proxy)?
WITH ins AS (SELECT dst AS account_id, step, amount FROM transactions WHERE is_self_transfer = 0),
     outs AS (SELECT src AS account_id, tx_id, step, amount FROM transactions WHERE is_self_transfer = 0),
     matched AS (
        SELECT o.account_id, o.tx_id, o.amount,
               MIN(o.step - i.step) AS hold_days
        FROM outs o
        JOIN ins i ON i.account_id = o.account_id AND i.step <= o.step AND i.step >= o.step - 3
        GROUP BY o.account_id, o.tx_id, o.amount
     )
SELECT m.account_id, COUNT(*) AS forwarded_outflows, ROUND(SUM(m.amount), 2) AS forwarded_value,
       ROUND(AVG(m.hold_days), 2) AS avg_hold_days,
       (SELECT COUNT(*) FROM transactions t WHERE t.src = m.account_id) AS all_outflows
FROM matched m
GROUP BY m.account_id
HAVING forwarded_outflows >= 5
ORDER BY forwarded_outflows DESC
LIMIT 25;

-- name: q09_alert_volume_by_rule
-- Q: How many alerts does each rule generate per period, and per 1,000 transactions?
WITH per_period AS (
    SELECT period,
           COUNT(*) AS account_weeks,
           SUM(sig_TXN_01) AS "TXN-01", SUM(sig_VEL_01) AS "VEL-01", SUM(sig_BEH_01) AS "BEH-01",
           SUM(sig_BEH_02) AS "BEH-02", SUM(sig_BEH_03) AS "BEH-03", SUM(sig_BEH_04) AS "BEH-04",
           SUM(sig_AML_01) AS "AML-01", SUM(sig_AML_02) AS "AML-02", SUM(sig_AML_03) AS "AML-03",
           SUM(sig_NET_01) AS "NET-01", SUM(sig_NET_02) AS "NET-02", SUM(sig_NET_03) AS "NET-03",
           SUM(alert) AS integrated_alerts
    FROM account_week_scores GROUP BY period
),
tx AS (
    SELECT CASE WHEN week BETWEEN 5 AND 12 THEN 'dev' WHEN week BETWEEN 13 AND 21 THEN 'test' END AS period,
           COUNT(*) AS transactions
    FROM transactions WHERE week BETWEEN 5 AND 21 GROUP BY 1
)
SELECT p.*, tx.transactions,
       ROUND(1000.0 * p.integrated_alerts / tx.transactions, 2) AS integrated_alerts_per_1000_tx,
       ROUND(1000.0 * (p."TXN-01") / tx.transactions, 2)       AS amount_rule_alerts_per_1000_tx
FROM per_period p JOIN tx USING (period)
ORDER BY period;

-- name: q10_alert_workload_by_week_and_tier
-- Q: What would the weekly review queue look like, and what share is high priority?
SELECT week,
       COUNT(*)                                                         AS alerts,
       SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END)         AS critical,
       SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END)             AS high,
       SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END)           AS medium,
       SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END)              AS low,
       ROUND(100.0 * SUM(CASE WHEN risk_level IN ('CRITICAL', 'HIGH') THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                        AS high_priority_pct,
       ROUND(AVG(risk_score), 2)                                        AS avg_score
FROM alert_queue
GROUP BY week
ORDER BY week;

-- name: q11_value_touching_alerted_accounts
-- Q: What share of test-period value involves an account alerted in the same week?
WITH alerted AS (SELECT DISTINCT account_id, week FROM alert_queue WHERE week BETWEEN 13 AND 21),
     t AS (SELECT * FROM transactions WHERE week BETWEEN 13 AND 21)
SELECT COUNT(*) AS test_transactions,
       SUM(CASE WHEN a1.account_id IS NOT NULL OR a2.account_id IS NOT NULL THEN 1 ELSE 0 END) AS touching_alerted,
       ROUND(100.0 * SUM(CASE WHEN a1.account_id IS NOT NULL OR a2.account_id IS NOT NULL THEN amount ELSE 0 END)
             / SUM(amount), 2) AS pct_value_touching_alerted
FROM t
LEFT JOIN alerted a1 ON a1.account_id = t.src AND a1.week = t.week
LEFT JOIN alerted a2 ON a2.account_id = t.dst AND a2.week = t.week;

-- name: q12_fan_in_hubs
-- Q: Which accounts collect from the most distinct senders (many-to-one) in the test weeks?
WITH fin AS (
    SELECT dst AS account_id, COUNT(DISTINCT src) AS distinct_senders, COUNT(*) AS inbound_tx,
           ROUND(SUM(amount), 2) AS inbound_value
    FROM transactions WHERE week BETWEEN 13 AND 21 AND is_self_transfer = 0 GROUP BY dst
),
fout AS (
    SELECT src AS account_id, COUNT(DISTINCT dst) AS distinct_receivers, ROUND(SUM(amount), 2) AS outbound_value
    FROM transactions WHERE week BETWEEN 13 AND 21 AND is_self_transfer = 0 GROUP BY src
)
SELECT fin.account_id, distinct_senders, inbound_tx, inbound_value,
       COALESCE(distinct_receivers, 0) AS distinct_receivers, COALESCE(outbound_value, 0) AS outbound_value,
       ROUND(COALESCE(outbound_value, 0) / inbound_value, 3) AS out_in_ratio,
       DENSE_RANK() OVER (ORDER BY distinct_senders DESC) AS fan_in_rank
FROM fin LEFT JOIN fout USING (account_id)
ORDER BY distinct_senders DESC
LIMIT 25;

-- name: q13_reciprocal_relationships
-- Q: How common are two-way (A->B and B->A) relationships - the simplest circular flow?
WITH pair AS (SELECT DISTINCT src, dst FROM transactions WHERE is_self_transfer = 0)
SELECT COUNT(*)                                                    AS directed_pairs,
       SUM(CASE WHEN p2.src IS NOT NULL THEN 1 ELSE 0 END)         AS pairs_with_reverse,
       ROUND(100.0 * SUM(CASE WHEN p2.src IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS reciprocity_pct
FROM pair p1
LEFT JOIN pair p2 ON p2.src = p1.dst AND p2.dst = p1.src;

-- name: q14_alert_label_overlap_by_tier
-- [EVALUATION] Q: How often do alerts in each tier fall on labelled accounts (held-out group, test weeks)?
SELECT q.risk_level,
       COUNT(*)                                  AS alerts,
       SUM(a.is_suspicious)                      AS on_labelled_accounts,
       ROUND(1.0 * SUM(a.is_suspicious) / COUNT(*), 3) AS precision
FROM alert_queue q
JOIN accounts a       ON a.account_id = q.account_id
JOIN account_groups g ON g.account_id = q.account_id
WHERE q.week BETWEEN 13 AND 21 AND g."group" = 1
GROUP BY q.risk_level
ORDER BY CASE q.risk_level WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END;

-- name: q15_labelled_vs_unlabelled_activity
-- [EVALUATION] Q: Do labelled accounts transact differently (count, amount, counterparties)?
WITH legs AS (
    SELECT src AS account_id, dst AS cp, amount FROM transactions WHERE is_self_transfer = 0
    UNION ALL
    SELECT dst, src, amount FROM transactions WHERE is_self_transfer = 0
),
acc AS (
    SELECT account_id, COUNT(*) AS n, AVG(amount) AS avg_amount, COUNT(DISTINCT cp) AS cps
    FROM legs GROUP BY account_id
)
SELECT a.is_suspicious, COUNT(*) AS accounts, ROUND(AVG(n), 2) AS avg_transactions,
       ROUND(AVG(avg_amount), 2) AS avg_amount, ROUND(AVG(cps), 2) AS avg_counterparties,
       ROUND(AVG(1.0 * n / cps), 3) AS avg_tx_per_counterparty
FROM acc JOIN accounts a USING (account_id)
GROUP BY a.is_suspicious;
