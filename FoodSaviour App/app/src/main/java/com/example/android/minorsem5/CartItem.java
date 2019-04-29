package com.example.android.minorsem5;

public class CartItem{
    private String item_details;

    public CartItem() {
    }

    public CartItem(String item_details) {
        this.item_details = item_details;
    }

    public String getItem_details() {
        return item_details;
    }

    public void setItem_details(String item_details) {
        this.item_details = item_details;
    }
}
