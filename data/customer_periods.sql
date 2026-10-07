-- Выгрузка из базы проекта retail-rfm-cohort-analysis (таблица sales — очищенные транзакции UCI Online Retail II).
-- Для каждого клиента: выручка и заказы за «предпериод» (дек 2009 — ноя 2010) и «период теста» (дек 2010 — ноя 2011).
SELECT customer_id,
       COALESCE(SUM(revenue) FILTER (WHERE invoice_date <  '2010-12-01'), 0)                           AS pre_revenue,
       COUNT(DISTINCT invoice) FILTER (WHERE invoice_date <  '2010-12-01')                             AS pre_orders,
       COALESCE(SUM(revenue) FILTER (WHERE invoice_date >= '2010-12-01' AND invoice_date < '2011-12-01'), 0) AS revenue,
       COUNT(DISTINCT invoice) FILTER (WHERE invoice_date >= '2010-12-01' AND invoice_date < '2011-12-01')   AS orders
FROM sales
WHERE invoice_date < '2011-12-01'
GROUP BY customer_id
HAVING COUNT(*) FILTER (WHERE invoice_date < '2010-12-01') > 0;   -- только клиенты, которые были до «эксперимента»
