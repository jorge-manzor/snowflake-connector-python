-- holiday_basic.sql
-- Basic query to retrieve holiday data

SELECT 
    fecha_feriado,
    nombre,
    tipo,
    irrenunciable
FROM 
    {database}.{schema}.{table}
WHERE 
    fecha_feriado >= {min_date}
ORDER BY 
    fecha_feriado