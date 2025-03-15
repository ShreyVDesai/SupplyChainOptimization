
```mermaid
erDiagram
    PRODUCT {
        VARCHAR product_name PK "Product Name"
    }
    SALES {
        DATE date "Date"
        VARCHAR product_name FK "Product Name"
        INT total_quantity "Total Quantity"
    }
    
    SALES }|..|{ PRODUCT : "references"

```