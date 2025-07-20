-- the five most sold products by Total revenue

select p.productname ,sum(p.price*o.quantity) as "revenue",c.categoryname as "Product Category" from products p 
inner join orderdetails o on p.productid = o.productid 
inner join categories c on p.categoryid =c.categoryid 
group by p.productname,c.categoryname 
order by revenue desc
limit 5


--- categories by month 
SELECT 
    c.categoryname,
    EXTRACT(MONTH FROM o2.orderdate) AS "month",
    SUM(o.quantity * p.price) AS total_revenue
FROM categories c
INNER JOIN products p ON p.categoryid = c.categoryid
INNER JOIN orderdetails o ON o.productid = p.productid
INNER JOIN orders o2 ON o.orderid = o2.orderid
GROUP BY c.categoryname, EXTRACT(MONTH FROM o2.orderdate)
ORDER BY c.categoryname, "month";

----employee by number of sales , quantity of products sold , revenue

select e.lastname,
e.firstname,
count(distinct o.orderid) as "number of sales",
sum(o2.quantity) as "quantity of products sold",
sum(p.price*o2.quantity) as "revenue by employee"
from employees e 
inner join orders o on e.employeeid = o.employeeid 
inner join orderdetails o2 on o.orderid = o2.orderid
inner join products p on p.productid = o2.productid 
group by e.lastname ,e.firstname 
order by "revenue by employee" desc



    ---Rank clients by spending
	select cs.customername,
	sum(p.price*o2.quantity) as "spending by client",
	RANK() over (order by sum(p.price*o2.quantity) desc) as rank 
	from customers cs
	inner join orders o on o.customerid = cs.customerid
	inner join orderdetails o2 on o2.orderid = o.orderid 
	inner join products p on p.productid = o2.productid 
	group by customername 
	
	
--- revenue by categories 

select c.categoryname,sum(p.price*o.quantity) as "revenue" from categories c 
inner join products p on p.categoryid = c.categoryid 
inner join orderdetails o on p.productid =o.productid 
group by c.categoryname 
order by "revenue" desc

	
---Average products per order and revenue per order

with 
cte_1 as (
    select 
        o.orderid,
        avg(o2.quantity) as "products by order" 
    from orders o
    inner join orderdetails o2 on o.orderid = o2.orderid 
    group by o.orderid
), 
cte_2 as (
    select 
        o.orderid,
        sum(p.price * o2.quantity) as "revenue by order"  
    from orders o 
    inner join orderdetails o2 on o.orderid = o2.orderid 
    inner join products p on p.productid = o2.productid
    group by o.orderid
)
select 
    round(avg(c1."products by order")) as "avg products per order",
    avg(c2."revenue by order") as "avg revenue per order"
from cte_1 c1
inner join cte_2 c2 on c1.orderid = c2.orderid;



-- supplier and amount of products and revenue  

select s.suppliername,count(distinct p.productid)as "amount of products", sum(p.price * o.quantity) as "revenue by supplier"from suppliers s 
inner join products p  on p.supplierid = s.supplierid 
inner join orderdetails o  on p.productid = o.productid 
group by s.suppliername 


--- month with the most revenue and sales 

SELECT EXTRACT(MONTH FROM o.orderdate) AS "month",sum(p.price * od.quantity) as "revenue by month",count(distinct o.orderid) as "sales number" FROM orders o
left join orderdetails  od on o.orderid = od.orderid
inner join products p on p.productid = od.productid
group by EXTRACT(MONTH FROM o.orderdate)


---employees with clients 

select e.firstname,e.lastname,count(distinct s.customerid) as "number of distinct costumers" from employees e 
inner join orders o on e.employeeid = o.employeeid 
inner join customers s on s.customerid= o.customerid 
group by  e.firstname,e.lastname



--- contribution of each product % to its category  

WITH revenue_by_product AS (
    SELECT 
        p.productname,
        c.categoryname,
        SUM(p.price * od.quantity) AS product_revenue
    FROM products p
    INNER JOIN categories c ON p.categoryid = c.categoryid
    INNER JOIN orderdetails od ON od.productid = p.productid
    GROUP BY p.productid, p.productname, c.categoryname
),

revenue_by_category AS (
    SELECT 
        c.categoryname,
        SUM(p.price * od.quantity) AS category_revenue
    FROM products p
    INNER JOIN categories c ON p.categoryid = c.categoryid
    INNER JOIN orderdetails od ON od.productid = p.productid
    GROUP BY c.categoryname
)

select rp.categoryname,rp.productname,round((rp.product_revenue/rv.category_revenue)*100,2) as "% of contribuition" from revenue_by_product rp
inner join revenue_by_category rv on rp.categoryname = rv.categoryname
order by rp.categoryname,"% of contribuition" desc












