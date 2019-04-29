package com.example.android.minorsem5;

public class UserProfile {
    String address="",myname="",email="",password="";

    public UserProfile(){};

    public UserProfile(String add,String name,String email,String pass) {
        this.address=add;
        this.myname=name;
        this.email=email;
        this.password=pass;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getMyname() {
        return myname;
    }

    public void setMyname(String myname) {
        this.myname = myname;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}

