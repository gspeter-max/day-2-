''' problem on leetcode '''
''' 1193. Monthly Transactions I (medium) '''
with temp_temp as (
    select *, date_format(trans_date,'%Y-%m') as month
    from transactions
) 
select month, country, count(id) as trans_count, 
sum(case when state = 'approved' then 1 else 0 end )as approved_count ,
sum(amount) as trans_total_amount , 
sum(case when state = 'approved' then amount else 0 end) as approved_total_amount
from temp_temp
group by month, country; 


'''1204. Last Person to Fit in the Bus (medium) problem '''
  
WITH temp_temp AS (
    SELECT 
        person_name,
        weight,
        SUM(weight) OVER (ORDER BY turn) AS total_weight,
        turn,  -- Include the 'turn' column here
        ROW_NUMBER() OVER (ORDER BY turn) AS row_n
    FROM queue
),
temp_temp_2 AS (
    SELECT *,
           CASE 
               WHEN total_weight <= 1000 THEN turn
               ELSE NULL
           END AS valid_turn
    FROM temp_temp
)
SELECT person_name
FROM temp_temp_2
WHERE valid_turn IS NOT NULL
ORDER BY valid_turn DESC
limit  1; 

''' 1211. Queries Quality and Percentage ''' 
  
select query_name , 
round(((sum(rating / position)) / count(*)),2)  as quality,
round((sum(case when rating < 3 then 1 else 0 end) / count(*)) * 100,2) as poor_query_percentage
from queries 
group by query_name ; 
