package com.example.android.minorsem5;

public class ProductItem {
    private String Product_name;
    private String expiry_date;
    private double product_discounted_price;
    private String product_image;
    private double product_price;
    private int quantity;
    //private String Product_Mfg;
    //private String Product_Exp;
    //private double Stores_Rating;


    public ProductItem(String product_name, String expiry_date, double product_discounted_price, String product_image, double product_price, int quantity) {
        Product_name = product_name;
        this.expiry_date = expiry_date;
        this.product_discounted_price = product_discounted_price;
        this.product_image = product_image;
        this.product_price = product_price;
        this.quantity = quantity;
    }

    public ProductItem() {
    }

    public String getProduct_name() {
        return Product_name;
    }

    public void setProduct_name(String product_name) {
        Product_name = product_name;
    }

    public String getExpiry_date() {
        return expiry_date;
    }

    public void setExpiry_date(String expiry_date) {
        this.expiry_date = expiry_date;
    }

    public double getProduct_discounted_price() {
        return product_discounted_price;
    }

    public void setProduct_discounted_price(double product_discounted_price) {
        this.product_discounted_price = product_discounted_price;
    }

    public String getProduct_image() {
        return product_image;
    }

    public void setProduct_image(String product_image) {
        this.product_image = product_image;
    }

    public double getProduct_price() {
        return product_price;
    }

    public void setProduct_price(double product_price) {
        this.product_price = product_price;
    }

    public int getQuantity() {
        return quantity;
    }

    public void setQuantity(int quantity) {
        this.quantity = quantity;
    }
}
