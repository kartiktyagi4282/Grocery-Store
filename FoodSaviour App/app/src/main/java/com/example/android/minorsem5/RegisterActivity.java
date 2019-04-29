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
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.HashMap;

public class RegisterActivity extends AppCompatActivity {

    private FirebaseAuth mAuth;
    private DatabaseReference mDatabase;
    private EditText name;
    private EditText email;
    private EditText address;
    private EditText password;
    private Button mCreateBtn;
    private Toolbar mtoolbar;
    private ProgressDialog mRegisterprogress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        // setting toolbar
        mtoolbar = (Toolbar) findViewById(R.id.register_app_bar);
        setSupportActionBar(mtoolbar);
        getSupportActionBar().setTitle("Create Account");
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        // firebase authentication
        mAuth = FirebaseAuth.getInstance();

        // set variables to android field
        name = (EditText)findViewById(R.id.reg_name);
        email =(EditText)findViewById(R.id.reg_email);
        address = (EditText)findViewById(R.id.reg_address);
        password = (EditText)findViewById(R.id.reg_password);
        mCreateBtn =(Button)findViewById(R.id.reg_createBtn);
        mCreateBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                register(name,email,address,password);
            }
        });
    }

    // function to register new user
    private void register(EditText name, final EditText email, EditText address, EditText password) {

        final String mName = name.getText().toString();
        final String mEmail = email.getText().toString();
        final String mAddress = address.getText().toString();
        final String mPassword = password.getText().toString();

        if (!mName.isEmpty() && !mEmail.isEmpty() && !mAddress.isEmpty() && !mPassword.isEmpty()) {
            // set the progress bar
            mRegisterprogress = new ProgressDialog(this);
            mRegisterprogress.setTitle("Signing In");
            mRegisterprogress.setMessage("Please wait while we create your account !!");
            mRegisterprogress.setCanceledOnTouchOutside(false);
            mRegisterprogress.show();

            // checking the user email and password
            mAuth.createUserWithEmailAndPassword(mEmail, mPassword).addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
                @Override
                public void onComplete(@NonNull Task<AuthResult> task) {
                    if (task.isSuccessful()) {

                        FirebaseUser currentUser = mAuth.getCurrentUser();
                        String userId = currentUser.getUid();
                        mDatabase = FirebaseDatabase.getInstance().getReference().child("User").child(userId);
                        HashMap<String,String>userInfo = new HashMap<>();
                        userInfo.put("Name", mName);
                        userInfo.put("Email", mEmail);
                        userInfo.put("Address", mAddress);
                        userInfo.put("Password", mPassword);
                        mDatabase.setValue(userInfo).addOnCompleteListener(new OnCompleteListener<Void>() {
                            @Override
                            public void onComplete(@NonNull Task<Void> task) {
                                if(task.isSuccessful()){
                                    mRegisterprogress.dismiss();
                                    Toast.makeText(RegisterActivity.this, "Successfully Register", Toast.LENGTH_SHORT).show();
                                    Intent mainIntent = new Intent(RegisterActivity.this, MainActivity.class);
                                    mainIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK);
                                    startActivity(mainIntent);
                                    finish();
                                }
                            }
                        });
                    }
                    else {
                        mRegisterprogress.hide();
                        Log.i("signInWithEmail:failed", String.valueOf(task.getException()));
                        Toast.makeText(RegisterActivity.this, "User Authentication Failed: " + task.getException().getMessage(), Toast.LENGTH_SHORT).show();
                    }
                }
            });
        }
        else{
            // if fields are empty
            Toast.makeText(RegisterActivity.this, "Empty Fields", Toast.LENGTH_SHORT).show();
        }
    }
}
