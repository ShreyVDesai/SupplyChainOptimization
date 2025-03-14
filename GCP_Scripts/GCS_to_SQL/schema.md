
```mermaid
erDiagram
    PRODUCT {
        VARCHAR product_name PK "Product Name"
    }
    TIME_DIMENSION {
        DATE date PK "Date"
        INT day_of_week "Day of Week"
        BOOLEAN is_weekend "Is Weekend"
        INT day_of_month "Day of Month"
        INT day_of_year "Day of Year"
        INT month "Month"
        INT week_of_year "Week of Year"
    }
    SALES {
        DATE date FK "Date"
        VARCHAR product_name FK "Product Name"
        INT total_quantity "Total Quantity"
    }
    
    SALES }|..|{ TIME_DIMENSION : "references"
    SALES }|..|{ PRODUCT : "references"
```