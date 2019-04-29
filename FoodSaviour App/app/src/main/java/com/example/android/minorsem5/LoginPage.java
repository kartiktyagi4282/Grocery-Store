package com.example.android.minorsem5;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;

public class LoginPage extends AppCompatActivity {

    private FirebaseAuth mAuth;
    private EditText email;
    private EditText password;
    private Button mloginBtn;
    private Toolbar mtoolbar;
    private ProgressDialog mLoginProgress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login_page);

        // setting toolbar
        mtoolbar = (Toolbar) findViewById(R.id.login_app_bar);
        setSupportActionBar(mtoolbar);
        getSupportActionBar().setTitle("Login");
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        // firebase authentication
        mAuth = FirebaseAuth.getInstance();

        // set variables to android field
        email = (EditText)findViewById(R.id.login_email);
        password = (EditText)findViewById(R.id.login_password);
        mloginBtn = (Button)findViewById(R.id.login_loginBtn);
        mloginBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                login_account(email,password);
            }
        });
    }

    public void login_account( EditText email, EditText password) {
        String mEmail = email.getText().toString();
        String mPassword = password.getText().toString();
        // if user fill the login credentials
        if(!mEmail.isEmpty() && !mPassword.isEmpty()) {

            mLoginProgress = new ProgressDialog(this);
            mLoginProgress.setTitle("Logging In");
            mLoginProgress.setMessage("Please wait while we check your credentials !!");
            mLoginProgress.setCanceledOnTouchOutside(false);
            mLoginProgress.show();
            mAuth.signInWithEmailAndPassword(mEmail, mPassword).addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
                @Override
                public void onComplete(@NonNull Task<AuthResult> task) {
                    if (task.isSuccessful()) {
                        mLoginProgress.dismiss();
                        Toast.makeText(LoginPage.this, "Successfully Login", Toast.LENGTH_SHORT).show();
                        Intent mainIntent = new Intent(LoginPage.this, MainActivity.class);
                        mainIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK);
                        startActivity(mainIntent);
                        finish();
                    } else {
                        mLoginProgress.hide();
                        //String error = "Sorry!!\nSome error occur";
                        Log.i("signInWithEmail:failed", String.valueOf(task.getException()));
                        Toast.makeText(LoginPage.this, "User Authentication Failed: " + task.getException().getMessage(), Toast.LENGTH_SHORT).show();
                    }
                }
            });
        }
        // if login fields are empty
        else{

            Toast.makeText(LoginPage.this, "Empty Fields", Toast.LENGTH_SHORT).show();
        }
    }
}
