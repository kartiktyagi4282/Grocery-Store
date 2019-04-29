package com.example.android.minorsem5;

import android.app.Application;
import com.firebase.client.Firebase;


public class user_firebase extends Application {

    @Override
    public void onCreate() {
        super.onCreate();


        Firebase.setAndroidContext(this);
    }

}
