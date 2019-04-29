package com.example.android.minorsem5;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class StartActivity extends AppCompatActivity {

    private Button button_addAccount;
    private Button button_login;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
        button_addAccount = (Button)findViewById(R.id.start_add_account_btn);
        button_addAccount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i("Start Activity","register request");
                Intent regIntent = new Intent(StartActivity.this,RegisterActivity.class);
                startActivity(regIntent);
            }
        });
        button_login = (Button)findViewById(R.id.start_login_btn);

        button_login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i("Start Activity","login request");
                Intent loginIntent = new Intent(StartActivity.this,LoginPage.class);
                startActivity(loginIntent);
            }
        });

    }
}
