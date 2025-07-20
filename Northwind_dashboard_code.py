import pandas as pd 
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm 
import numpy as np
import streamlit as st


# Configuración de la página
st.set_page_config(page_title="Northwind Company", page_icon='🌐', layout="wide")
st.title("🌐 Northwind Company 💸💳🧾💼")

with st.expander("🔌 Configuración de conexión a la base de datos"):
    db_url = st.text_input("URL de conexión a PostgreSQL:", 
                         value="postgresql://neondb_owner:dVO76wDFuhWM@ep-bitter-pond-a5neo6us.us-east-2.aws.neon.tech/neondb?sslmode=require")
    schema = st.text_input("Esquema de la base de datos:", value="northwind")

if db_url:
    try:
        engine = create_engine(f"{db_url}&options=-csearch_path%3D{schema}")
        st.success("✅ Conexión a la base de datos establecida correctamente")
        
        # productos overview
        query_products = '''select p.productname ,sum(p.price*o.quantity) as "revenue",c.categoryname as "Product Category" from products p 
        inner join orderdetails o on p.productid = o.productid 
        inner join categories c on p.categoryid =c.categoryid 
        group by p.productname,c.categoryname 
        order by revenue desc
        '''
        query_products= pd.read_sql(query_products, engine)
        
        # Colores por categoría
        categories = query_products['Product Category'].unique()
        palette = sns.color_palette("pastel", len(categories)).as_hex()
        category_colors = dict(zip(categories, palette))

        # Crear nueva columna con celda coloreada
        query_products['Categoria con color'] = query_products['Product Category'].apply(
            lambda cat: f'<div style="background-color: {category_colors[cat]}; padding: 4px;">{cat}</div>'
        )

        # Mostrar tabla con st.markdown
        with st.expander("Products --- table"):
            st.markdown("### Productos con categoría coloreada")
            st.markdown(
                query_products[['productname', 'revenue', 'Categoria con color']].to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        


        
        
        # 1. Los 5 productos más rentables según margen de ganancia
        query1 = '''
        SELECT p.productname, sum(p.price*o.quantity) as "revenue", 
               c.categoryname as "Product Category" 
        FROM products p 
        INNER JOIN orderdetails o ON p.productid = o.productid 
        INNER JOIN categories c ON p.categoryid = c.categoryid 
        GROUP BY p.productname, c.categoryname 
        ORDER BY revenue DESC
        LIMIT 5
        '''    
        five_most_sold = pd.read_sql(query1, engine)
        
        
        with st.expander("📊 Los 5 productos más rentables según margen de ganancia"):
            # Dividir en dos columnas (60% gráfico, 40% tabla)
            col1, col2 = st.columns([6, 4])
            
            with col1:
                # Crear y mostrar el gráfico
                fig_1 = px.bar(
                    five_most_sold,
                    x='productname',
                    y='revenue',
                    color='Product Category',
                    title='Top 5 Productos con Mayor Revenue',
                    labels={'productname': 'Producto', 'revenue': 'Revenue ($)'},
                    text_auto='.2s',
                    height=500
                )
                fig_1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=-45,
                    hovermode="x unified"
                )
                fig_1.update_traces(
                    textfont_size=12,
                    textangle=0,
                    textposition="outside",
                    cliponaxis=False
                )
                st.plotly_chart(fig_1, use_container_width=True)
            
            with col2:
                # Mostrar la tabla con formato
                        # Espaciado superior para alinear con el gráfico
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.write("")  # Espacio vacío
                st.dataframe(
                    five_most_sold.style
                    .format({'revenue': '${:,.2f}'})
                    .set_properties(**{'background-color':  "#DAA2D7"}),
                    height=213
                )
                
            # KPIs debajo
            st.subheader("KPIs")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Producto #1", 
                         value=five_most_sold.iloc[0]['productname'],
                         delta=f"${five_most_sold.iloc[0]['revenue']:,.2f}")
            with cols[1]:
                st.metric("Revenue Total top 5 ", 
                         value=f"${five_most_sold['revenue'].sum():,.2f}")
            with cols[2]:
                st.metric("Categoría Líder",
                         value=five_most_sold.iloc[0]['Product Category'])
                
         # 2. Desempeño por empleado
        query_2 = '''select e.lastname,
            e.firstname,
            count(distinct o.orderid) as "number of sales",
            sum(o2.quantity) as "quantity of products sold",
            sum(p.price*o2.quantity) as "revenue by employee"
            from employees e 
            inner join orders o on e.employeeid = o.employeeid 
            inner join orderdetails o2 on o.orderid = o2.orderid
            inner join products p on p.productid = o2.productid 
            group by e.lastname ,e.firstname 
            order by "revenue by employee" desc'''
        employee_overview = pd.read_sql(query_2, engine)
        employee_overview.rename(columns={'revenue by employee':'revenue'}, inplace=True)
        
        with st.expander("🪪 Desempeño de los empleados"):
            col1, col2 = st.columns([5,5])
            
            with col1:
                # Prepara los datos
                employee_overview['employee_name'] = employee_overview['firstname'] + ' ' + employee_overview['lastname']
                
                # Selector de métrica
                selected_metric = st.selectbox(
                    "Selecciona la métrica a visualizar:",
                    options=['number of sales', 'quantity of products sold', 'revenue'],
                    format_func=lambda x: x.title(),
                    key='employee_metric'
                )
                
                # Gráfico con Plotly Express
                fig = px.scatter(
                    employee_overview,
                    x='employee_name',
                    y=selected_metric,
                    size='revenue',
                    color='quantity of products sold',
                    color_continuous_scale='Viridis',
                    title=f"Desempeño por Empleado - {selected_metric.title()}",
                    size_max=40,
                    hover_data=['number of sales', 'quantity of products sold', 'revenue'],
                    labels={
                        'employee_name': 'Empleado',
                        'quantity of products sold': 'Cantidad Vendida'
                    }
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig)
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                # Mostrar dataframe con formato mejorado
                st.dataframe(
                    employee_overview.style
                    .format({'revenue': '${:,.2f}'})
                    .set_properties(**{'background-color':  "#DAA2D7"}),
                    height=360
                )   
        # clientes más valiosos según su gasto total en la empresa  
        query_3 = '''select cs.customername,
            sum(p.price*o2.quantity) as "spending by client",
            RANK() over (order by sum(p.price*o2.quantity) desc) as rank 
            from customers cs
            inner join orders o on o.customerid = cs.customerid
            inner join orderdetails o2 on o2.orderid = o.orderid 
            inner join products p on p.productid = o2.productid 
            group by customername '''
        Rank_clients_by_spending= pd.read_sql(query_3, engine)
        with st.expander("📈 clientes más valiosos según su gasto total en la empresa"):
            col1,col2=st.columns([6,4])
            with col1:
                num_clients = st.slider(
                    "Selecciona el número de clientes a mostrar:",
                    min_value=5,
                    max_value=73,
                    value=10,  # Valor por defecto
                    step=1,
                    help="Mueve el slider para ajustar el número de clientes visibles"
                )

                # 2. Filtrar datos
                df_filtered = Rank_clients_by_spending.head(num_clients)
                fig = px.scatter(
                    df_filtered,
                    x='customername',
                    y='spending by client',
                    color='rank',
                    color_continuous_scale='Plasma',
                    title=f"Clients by spending ranking",
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig)
                
            with col2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.dataframe(
                    Rank_clients_by_spending.style
                    .format({'revenue': '${:,.2f}'})
                    .set_properties(**{'background-color': "#DAA2D7"}),
                    height=300
                )
            # KPIs debajo
            st.subheader("KPIs")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Mejor Cliente", 
                         value=Rank_clients_by_spending.iloc[0]['customername'],
                         delta=f"${Rank_clients_by_spending.iloc[0]['spending by client']:,.2f}")
            with cols[1]:
                st.metric("2 Mejor Cliente", 
                         value=Rank_clients_by_spending.iloc[1]['customername'],
                         delta=f"${Rank_clients_by_spending.iloc[1]['spending by client']:,.2f}")
            with cols[2]:
                st.metric("3 Mejor Cliente",
                         value=Rank_clients_by_spending.iloc[2]['customername'],
                         delta=f"${Rank_clients_by_spending.iloc[2]['spending by client']:,.2f}")  
                
        # Ganancias por categoria de producto 
        
        query_3 = '''select c.categoryname,sum(p.price*o.quantity) as "revenue" from categories c 
                        inner join products p on p.categoryid = c.categoryid 
                        inner join orderdetails o on p.productid =o.productid 
                        group by c.categoryname 
                        order by "revenue" desc'''
        revenue_by_categories= pd.read_sql(query_3, engine)
        with st.expander("💰 Ganancias por categoria de producto "):
            col1,col2=st.columns([5,4])
            with col1:
                fig_4=px.pie(revenue_by_categories,names='categoryname',values='revenue',title='Revenue by category ')
                st.plotly_chart(fig_4)       
            
            with col2 :
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.dataframe(
                    revenue_by_categories.style
                    .format({'revenue': '${:,.2f}'})
                    .set_properties(**{'background-color': "#DAA2D7"}),
                    height=323)
                
            # KPI
            st.subheader("Kpis")
            total_revenue = revenue_by_categories['revenue'].sum()
            st.metric("Total de Ganancias Recaudadas", f"${total_revenue:,.2f}")            
                
        # mapa de calor  categorias por mes 
        query_categories='''SELECT 
            c.categoryname,
            EXTRACT(MONTH FROM o2.orderdate) AS "month",
            SUM(o.quantity * p.price) AS total_revenue
        FROM categories c
        INNER JOIN products p ON p.categoryid = c.categoryid
        INNER JOIN orderdetails o ON o.productid = p.productid
        INNER JOIN orders o2 ON o.orderid = o2.orderid
        GROUP BY c.categoryname, EXTRACT(MONTH FROM o2.orderdate)
        ORDER BY c.categoryname, "month";'''
        query_categories=pd.read_sql(query_categories, engine)
        
        with st.expander ("💴💵💶 Ganancias meses por categoria "):
            query_categories[['month','total_revenue']]=query_categories[['month','total_revenue']].astype(int)
            month_names = {
                    1: 'January',
                    2: 'February',
                    7: 'July',
                    8: 'August',
                    9: 'September',
                    10: 'October',
                    11: 'November',
                    12: 'December'
                }

                # Reemplazar los números por los nombres del mes
            query_categories['month'] =query_categories['month'].replace(month_names)
            query_categories=query_categories.pivot_table(index='categoryname',columns='month',values='total_revenue')
            imshow = px.imshow(
                query_categories,
                labels={'x': 'Mes', 'y': 'Categoría', 'color': 'Ingresos totales'},
                x=query_categories.columns,  
                y=query_categories.index,    
                aspect="auto",               # ajusta relación de aspecto
                color_continuous_scale="Plasma"  # paleta más visual
            )

            imshow.update_layout(
                    title="🟨 Mapa de calor: Ingresos por categoría y mes",
                    title_x=0.5,
                    width=900,
                    height=600,
                    margin=dict(l=60, r=60, t=80, b=80),
                    xaxis_title="Mes",
                    yaxis_title="Categoría",
                    font=dict(family="Arial", size=12),
                    coloraxis_colorbar=dict(
                        title="USD",
                        ticksuffix="$",
                        thickness=15,
                        len=0.75,
                        yanchor="middle",
                    )
                )
                
            st.plotly_chart(imshow, use_container_width=True)

        
        # boxplot ganancia 
        
        query_box = '''
                SELECT 
                    p.productname,
                    SUM(o.quantity * p.price) AS revenue,
                    c.categoryname
                FROM 
                    products p
                INNER JOIN 
                    orderdetails o ON p.productid = o.productid
                INNER JOIN 
                    categories c ON p.categoryid = c.categoryid
                GROUP BY 
                    p.productname, c.categoryname
                '''
        query_box = pd.read_sql(query_box, engine)
        
        with st.expander ("💀 Distribucion de ganancia por categoria "):
            fig = px.box(
                query_box,
                x='categoryname',
                y='revenue',
                color='categoryname',  
                title='Distribución del Revenue por Categoría de Producto',
                points='all',  
                color_discrete_sequence=px.colors.qualitative.Bold  
            )

            fig.update_layout(
                width=1200,  # Aumenta el ancho del gráfico
                height=800,
                #template='plotly_white',
                xaxis_title='Categoría de Producto',
                yaxis_title='Ganancia (USD)',
                title_font_size=22,
                xaxis_tickangle=-45,
                font=dict(family='Arial', size=14),
                showlegend=False
            )

            st.plotly_chart(fig,use_container_width=True)
            
        
            
                     
        # Supplier a detalle
        query_5 = '''select s.suppliername,count(distinct p.productid)as "amount of products", sum(p.price * o.quantity) as "revenue by supplier"from suppliers s 
                    inner join products p  on p.supplierid = s.supplierid 
                    inner join orderdetails o  on p.productid = o.productid 
                    group by s.suppliername'''
        supplier= pd.read_sql(query_5, engine)
        with st.expander("🔩⚙️Suppliers por productos individuales y ganancia total"):
            col1,col2=st.columns([6,4])
            
            with col1:
                fig_6 = px.bar(
                    supplier.sort_values('revenue by supplier', ascending=False),
                    x='suppliername', 
                    y='revenue by supplier',
                    color='amount of products',
                    title='Revenue by Supplier (Colored by Product Count)',
                    labels={'suppliername': 'Supplier', 'revenue by supplier': 'Revenue'},
                    color_continuous_scale='Viridis'
                )
                fig_6.update_layout(xaxis_title=None, xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_6, use_container_width=True)
                
            with col2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.dataframe(
                    supplier.style
                    .format({'revenue': '${:,.2f}'})
                    .set_properties(**{'background-color': "#DAA2D7"}),
                    height=300)
                
            st.subheader("⚗️ Supplier KPIs")
            cols = st.columns(3)

            with cols[0]:
                # KPI 1: Top Revenue Supplier con icono y formato mejorado
                top_supplier = supplier.loc[supplier['revenue by supplier'].idxmax()]
                st.metric(
                    label="🏆 Top Revenue Supplier", 
                    value=f"{top_supplier['suppliername'][:15]}..." if len(top_supplier['suppliername']) > 15 else top_supplier['suppliername'],
                    delta=f"${top_supplier['revenue by supplier']:,.2f}",
                    help=f"Full name: {top_supplier['suppliername']}"
                )

            with cols[1]:
                # KPI 2: Average Products con icono, redondeo y formato mejorado
                avg_products = round(supplier['amount of products'].mean())
                st.metric(
                    label="📦 Avg Products/Supplier", 
                    value=f"{avg_products}",
                    delta=f"{int(round(avg_products))} products avg",
                    help="Average number of distinct products per supplier"
                )

            with cols[2]:
                # KPI 3: Total Revenue con formato de moneda
                total_revenue = supplier['revenue by supplier'].sum()
                st.metric(
                    label="💰 Total Supplier Revenue", 
                    value=f"${total_revenue:,.0f}",
                    delta=f"From {len(supplier)} suppliers",
                    help="Sum of all revenue generated by suppliers"
                )
                                
                        
        # Ganancias por mes
        query_6 = '''SELECT EXTRACT(MONTH FROM o.orderdate) AS "month",sum(p.price * od.quantity) as "revenue by month",count(distinct o.orderid) as "sales number" FROM orders o
        left join orderdetails  od on o.orderid = od.orderid
        inner join products p on p.productid = od.productid
        group by EXTRACT(MONTH FROM o.orderdate)'''
        month_revenue_sales= pd.read_sql(query_6, engine)
        # Diccionario de número a nombre de mes
        month_names = {
            1: 'January',
            2: 'February',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December'
        }

        # Reemplazar los números por los nombres del mes
        month_revenue_sales['month'] = month_revenue_sales['month'].replace(month_names)   
        
        with st.expander("📅⏳ Por Mes ganancias y ventas"):
            # Crear el treemap
            fig= px.treemap(
                month_revenue_sales,
                path=['month'],                     # Jerarquía (meses)
                values='revenue by month',          # Tamaño según ingresos
                color='sales number',               # Color según número de ventas (mejor contraste)
                color_continuous_scale='Tealrose',  # Escala de colores profesional
                title='<b>Revenue and Sales by Month</b>',  # Título en negrita
                hover_data={'sales number': True, 'revenue by month': ':.0f'},  # Formato hover
                labels={'sales number': 'Ventas', 'revenue by month': 'Ingresos (USD)'}
            )

            # Ajustes de diseño profesional
            fig.update_layout(
                margin=dict(l=0, r=0, t=50, b=0),  # Márgenes ajustados
                paper_bgcolor='white',              # Fondo blanco
                plot_bgcolor='white',               # Área del gráfico blanca
                font=dict(family='Arial', size=12), # Fuente profesional
                title_x=0.5,                        # Título centrado
            )

            # Quitar bordes de los bloques
            fig.update_traces(
                marker=dict(line=dict(width=0)),    # Sin bordes entre bloques
                textinfo='label+value',             # Muestra mes + valor
                texttemplate='<b>%{label}</b><br>%{value:,.0f} USD',  # Texto formateado
                hovertemplate='<b>%{label}</b><br>Ingresos: %{value:,.0f} USD<br>Ventas: %{color}'
            )

            st.plotly_chart(fig, use_container_width=True)                   
        
        
        # Numero de clientes individuales captados por empleado
        query_7 = '''select e.firstname,e.lastname,count(distinct s.customerid) as "number of distinct customers" from employees e 
                    inner join orders o on e.employeeid = o.employeeid 
                    inner join customers s on s.customerid= o.customerid 
                    group by  e.firstname,e.lastname'''
        employees= pd.read_sql(query_7, engine)
        employees=employees.sort_values(by='number of distinct customers',ascending=False)
        
        with st.expander("🔐🛡️ Numero de clientes individuales captados por empleado"):
            fig = go.Figure(go.Funnel(  
                y=employees['firstname'],
                x=employees['number of distinct customers'],
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.65,
                marker={
                    "color": ["deepskyblue", "lightsalmon", "tan", "teal", "silver","green","white","red","royalblue"],
                    "line": {
                        "width": [4, 2, 2, 3, 1, 1],
                        "color": ["wheat", "wheat", "blue", "wheat", "wheat"]
                    }
                },
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
            ))

            fig.update_layout(
                autosize=False,
                width=800,  # Ancho en píxeles
                height=600,  # Alto en píxeles
                margin=dict(l=50, r=50, b=50, t=50, pad=5)  # Márgenes
            )
            
            st.plotly_chart(fig)
     
        #---
        query_8 = '''WITH revenue_by_product AS (
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

                SELECT 
                    rp.categoryname,
                    rp.productname,
                    ROUND((rp.product_revenue/rv.category_revenue)*100,2) as percent_contrib 
                FROM revenue_by_product rp
                INNER JOIN revenue_by_category rv ON rp.categoryname = rv.categoryname
                ORDER BY rp.categoryname, percent_contrib DESC;'''

        contribuition_product_porcent = pd.read_sql(query_8, engine)

        with st.expander("📈 Contribución de cada producto en ganancia por categoría"):
            #  KPIs for top products by category
            st.subheader("🏆 Productos líderes por categoría")
            
            # Get top product for each category
            top_products = contribuition_product_porcent.loc[
                contribuition_product_porcent.groupby('categoryname')['percent_contrib'].idxmax()
            ]
            
            # Display in columns
            cols = st.columns(2)  
            for idx, row in top_products.iterrows():
                with cols[idx % len(cols)]:
                    st.metric(
                        label=f"Top {row['categoryname']}",
                        value=row['productname'],
                        delta=f"{row['percent_contrib']}% contribución",
                        help=f"Este producto aporta el {row['percent_contrib']}% de las ganancias en {row['categoryname']}"
                    )
            
            st.divider()
            
            # Then show the detailed table with all products
            st.subheader("📊 Detalle completo por categoría")
            
            # Creating a bar chart for each category
            for category in contribuition_product_porcent['categoryname'].unique():
                cat_data = contribuition_product_porcent[contribuition_product_porcent['categoryname'] == category]
                
                with st.expander(f"📌 {category}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Summary metrics for the category
                        top_product = cat_data.iloc[0]
                        st.metric(
                            "Producto líder",
                            top_product['productname'],
                            delta=f"{top_product['percent_contrib']}%"
                        )
                        avg_contrib = round(cat_data['percent_contrib'].mean(), 1)
                        st.metric(
                        f"Contribución  en {category}",
                        f"{avg_contrib}%",
                        help="Promedio de lo que cada producto aporta a las ventas de SU categoría"
                    )
                    
                    with col2:
                        # Interactive bar chart
                        fig = px.bar(
                            cat_data,
                            x='productname',
                            y='percent_contrib',
                            color='percent_contrib',
                            color_continuous_scale='tealrose',
                            title=f"Contribución por producto en {category}",
                            labels={'productname': 'Producto', 'percent_contrib': 'Contribución (%)'}
                        )
                        fig.update_layout(xaxis_title=None)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Optional: Show raw data table at the end
            st.dataframe(
                contribuition_product_porcent.style
                .background_gradient(subset=['percent_contrib'], cmap='YlGnBu')
                .format({'percent_contrib': '{:.2f}%'}),
                hide_index=True,
                use_container_width=True
            )
         
        # productos y ganancia por orden promedio ----- clientes a detalle    
        
        query_4 = '''with 
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
                inner join cte_2 c2 on c1.orderid = c2.orderid;'''
        per_order_average= pd.read_sql(query_4, engine)   
        per_order_average=per_order_average.astype(int)
        
        query_10 = '''select c.customername ,
                    sum(p.price*o2.quantity) as "total spent",
                    count(distinct o.orderid) as "number of orders",
                    sum(o2.quantity) as "number of products",
                    min (o.orderdate) as "first_order",
                    max (o.orderdate) as "last order",
                    count(distinct o.orderdate) as "unique days"
                    from customers c 
                    inner join orders o on c.customerid = o.customerid 
                    inner join orderdetails o2  on o2.orderid  = o.orderid 
                    inner join products p  on p.productid = o2.productid 
                    group by c.customername '''
        customer_information= pd.read_sql(query_10, engine)
        reference_date = customer_information['last order'].max()
        customer_information['recency_days'] = (reference_date - customer_information['last order']).dt.days  
        
        with st.expander("🤖🧬 productos y ganancia por orden promedio ----- clientes a detalle "):
            col1,col2=st.columns([1.8,5])
            with col1:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                per_order_average
            
            with col2:
                customer_information    
        
        
        # Segmentación de clientes usando K-Means

        # 1. Selección de características para el clustering
        X = customer_information[['total spent', 'number of orders', 'number of products', 'unique days', 'recency_days']]

        # 2. Estandarización de los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Aplicación del algoritmo K-Means
        kmeans = KMeans(n_clusters=3, random_state=0)
        customer_information['cluster'] = kmeans.fit_predict(X_scaled)

        # Visualización de resultados
        with st.expander("💸💳 Segmentación de Clientes", expanded=False):
            st.subheader("📊 Análisis de Segmentación")
            
            # 4. Gráfico Pairplot
            st.markdown("**Distribución por Clusters**")
            fig = sns.pairplot(
                customer_information, 
                vars=['total spent', 'number of orders', 'number of products', 'unique days', 'recency_days'], 
                hue='cluster',
                palette='viridis'
            )
            plt.suptitle("Relación entre Variables por Segmento", y=1.02)
            st.pyplot(fig.fig)
            
            # 5. Estadísticas por cluster
            st.subheader("📈 Características de Cada Segmento")
            clusteringcustomer = customer_information.groupby('cluster')[
                ['total spent', 'number of orders', 'number of products', 'unique days', 'recency_days']
            ].mean().sort_values('total spent', ascending=False)
            
            # Formateo y visualización de la tabla
            st.dataframe(
                clusteringcustomer.style
                .background_gradient(
                    subset=['total spent', 'number of orders', 'number of products'], 
                    cmap='YlGnBu'
                )
                .format({
                    'total spent': '${:,.2f}',
                    'number of orders': '{:.1f}',
                    'number of products': '{:.1f}',
                    'unique days': '{:.1f}',
                    'recency_days': '{:.1f} días'
                }),
                use_container_width=True
            )
            
            # 6. Interpretación de clusters
            st.subheader("🔍 Interpretación de Segmentos")
            st.markdown("""
            **Análisis de los clusters basado en el comportamiento de compra:**

            - **Cluster 1 (Clientes Premium)**  
            💎 **Gasto promedio**: $15,541.51  
            🛒 **Pedidos**: 5.69 por cliente  
            📦 **Productos distintos**: 516  
            📅 **Frecuencia de compra**: Compra cada 18 días en promedio  
            *Perfil*: Nuestros mejores clientes, con alto volumen de compras y máxima fidelidad.

            - **Cluster 2 (Clientes Regulares)**  
            💰 **Gasto promedio**: $3,718.98  
            🛒 **Pedidos**: 2.18 por cliente  
            📦 **Productos distintos**: 119  
            📅 **Frecuencia de compra**: Compra cada 43 días  
            *Perfil*: Clientes con potencial de crecimiento, buena frecuencia pero menor volumen.

            - **Cluster 0 (Clientes Ocasionales)**  
            💵 **Gasto promedio**: $1,783.39  
            🛒 **Pedidos**: 1.68 por cliente  
            📦 **Productos distintos**: 64  
            📅 **Frecuencia de compra**: Última compra hace 160 días  
            *Perfil*: Clientes inactivos o de bajo compromiso, requieren estrategias de reactivación.
            """)
            
        with st.expander("🔮 Modelo Predictivo: Total Gastado por Productos", expanded=True):
            st.subheader("Entrenamiento del Modelo Gradient Boosting")
            
            #customer_information=customer_information[customer_information['number of products']<300]
            
            # 1. Preparación de datos
            X = customer_information[['number of products']]
            y = customer_information['total spent']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 2. Configuración de parámetros para GridSearch
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [2, 3],
                'subsample': [1.0]
            }
            
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 3. Entrenamiento con GridSearchCV
            with st.spinner('Buscando los mejores hiperparámetros...'):
                gb = GradientBoostingRegressor(random_state=42)
                grid_search = GridSearchCV(
                    estimator=gb,
                    param_grid=param_grid,
                    scoring='r2',
                    cv=2,
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                progress_bar.progress(100)
                status_text.success('¡Modelo entrenado con éxito!')
            
            # 4. Mostrar resultados
            st.subheader("Resultados del Modelo")
            
            # Mejores parámetros en un dataframe
            best_params = pd.DataFrame.from_dict(grid_search.best_params_, orient='index', columns=['Valor'])
            st.dataframe(best_params)
            st.markdown("**Metricas del modelo**")
            # Evaluación del modelo
            best_gb = grid_search.best_estimator_
            y_pred = best_gb.predict(X_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("MAE", f"${mean_absolute_error(y_test, y_pred):.2f}")
            with col3:
                st.metric("MAPE", f"{mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
            
            # 5. Visualización de predicciones vs reales
            st.subheader("Predicciones vs Valores Reales")
            fig, ax = plt.subplots()
            ax.scatter(X_test['number of products'], y_test, color='blue', label='Datos Reales')
            ax.scatter(X_test['number of products'], y_pred, color='red', label='Predicciones', alpha=0.5)
            ax.set_xlabel('Número de Productos')
            ax.set_ylabel('Total Gastado')
            ax.legend()
            st.pyplot(fig)
        # Slider para predicción manual
        st.subheader("Simulador de Predicción")
        num_products = st.slider("Número de productos", 1, 20, 10)

            # Vector completo
        prediction = best_gb.predict([[num_products]])
        st.metric("Total gastado estimado", f"${prediction[0]:.2f}")   
            
            
                
        
                                             
    except Exception as e:
        st.error(f"❌ Error al conectar a la base de datos: {str(e)}")
else:
    st.warning("Por favor ingresa la URL de conexión a la base de datos")
    

        
        
 