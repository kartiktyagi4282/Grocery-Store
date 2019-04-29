package com.example.android.minorsem5;

//import android.support.v7.app.AppCompatActivity;

public class ListItem {

//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_list_item);
//    }

    private String Stores_name;
    private String Stores_Address;
    private String Stores_Contact;
    private String Stores_Image;
    private double Stores_Rating;

    public ListItem(String stores_name, String stores_Address, String stores_Contact, String stores_Image, double stores_Rating) {
        Stores_name = stores_name;
        Stores_Address = stores_Address;
        Stores_Contact = stores_Contact;
        Stores_Image = stores_Image;
        Stores_Rating = stores_Rating;
    }

    public ListItem(){

    }

    public String getStores_name() {
        return Stores_name;
    }

    public void setStores_name(String stores_name) {
        Stores_name = stores_name;
    }

    public String getStores_Address() {
        return Stores_Address;
    }

    public void setStores_Address(String stores_Address) {
        Stores_Address = stores_Address;
    }

    public String getStores_Contact() {
        return Stores_Contact;
    }

    public void setStores_Contact(String stores_Contact) {
        Stores_Contact = stores_Contact;
    }

    public String getStores_Image() {
        return Stores_Image;
    }

    public void setStores_Image(String stores_Image) {
        Stores_Image = stores_Image;
    }

    public double getStores_Rating() {
        return Stores_Rating;
    }

    public void setStores_Rating(double stores_Rating) {
        Stores_Rating = stores_Rating;
    }
}
