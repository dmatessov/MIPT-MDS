

--    1. Если сумма транзакций не более 50000, тогда вывести 'низкая доходность';
--    2. Если сумма транзакций больше 50000 и не более 10000 тогда вывести 'средняя доходность';
--    3. Если сумма транзакций больше 100000 тогда вывести 'высокая доходность'.
SELECT
	CUSTOMER_ID,
	case
		when not sum(TX_AMOUNT) > 5000 then 'низкая доходность'
		when sum(TX_AMOUNT) > 5000 and not sum(TX_AMOUNT) > 100000 then 'средняя доходность'
		when sum(TX_AMOUNT) > 100000 then 'высокая доходность'
	end plevel
FROM final_transactions
group by CUSTOMER_ID;
