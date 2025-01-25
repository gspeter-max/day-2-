''' Extremely Hard Problems for Practice'''

''' 
Problem 1: Recursive CTE for Hierarchical Data
You have the same employees table. Write a query using a recursive CTE to find the 
hierarchical structure of the organization, 
displaying emp_name along with all their managers up the hierarchy.
''' 
WITH RECURSIVE employees_hierarchy AS (
    -- Base Case: Start with employees with no manager
    SELECT emp_id,
           emp_name,
           manager_id,
           CAST(emp_name AS VARCHAR(1000)) AS hierarchy_path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive Case: Find employees reporting to the current level
    SELECT e.emp_id,
           e.emp_name,
           e.manager_id,
           CONCAT(h.hierarchy_path, '-->', e.emp_name) AS hierarchy_path
    FROM employees e
    INNER JOIN employees_hierarchy h ON e.manager_id = h.emp_id
)
SELECT emp_id, emp_name, hierarchy_path
FROM employees_hierarchy;



''' Problem 2: CTE for Running Totals '''
  '''
You have a sales table:
CREATE TABLE sales (
    sales_id INT,
    sale_date DATE,
    amount INT
);
Write a query to calculate the cumulative sales by day using a CTE, 
  ensuring the order is by sale_date.'''

  WITH RECURSIVE temp_temp AS (
    -- Base Case: Start with the first sale
    SELECT sales_id,
           sale_date,
           amount,
           CAST(amount AS INT) AS cumulative_sales
    FROM sales
    WHERE sales_id = 1

    UNION ALL

    -- Recursive Case: Add subsequent sales for the same sales_id in chronological order
    SELECT s.sales_id,
           s.sale_date,
           s.amount,
           t.cumulative_sales + s.amount AS cumulative_sales
    FROM sales s
    INNER JOIN temp_temp t 
        ON s.sales_id = t.sales_id AND s.sale_date > t.sale_date
)
SELECT sales_id,
       sale_date,
       amount,
       cumulative_sales
FROM temp_temp
ORDER BY sale_date;


'''Problem 3: Combining Multiple CTEs ''' 
  '''
Given two tables, students and exams:

CREATE TABLE students (
    student_id INT,
    student_name VARCHAR(100)
);

CREATE TABLE exams (
    exam_id INT,
    student_id INT,
    score INT
);
Write a query using multiple CTEs to:

Find the average score per student.
Rank students by their average score.
Display only the top 3 students, with their scores and ranks. ''' 

WITH temp_temp AS (
    SELECT
        e.student_id,
        s.student_name,
        e.score,
        e.exam_id
    FROM exams e
    LEFT JOIN students s ON e.student_id = s.student_id
),
temp_temp_2 AS (
    SELECT
        student_id,
        AVG(score) AS avg_score
    FROM temp_temp
    GROUP BY student_id
),
temp_temp_3 AS (
    SELECT 
        t1.student_id, 
        t1.student_name, 
        t2.avg_score,
        ROW_NUMBER() OVER (ORDER BY t2.avg_score DESC) AS row_n
    FROM temp_temp t1
    LEFT JOIN temp_temp_2 t2 ON t1.student_id = t2.student_id
)
SELECT
    student_id,
    student_name,
    avg_score,
    row_n
FROM temp_temp_3
WHERE row_n <= 3;

