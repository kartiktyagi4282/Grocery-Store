package com.example.android.minorsem5;

import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

public class UserInfo extends AppCompatActivity {

    private EditText pname,paddress,pmail,ppass;
    private Button ProfileUpdate;
    private FirebaseAuth firebaseauth;
    private FirebaseDatabase firebasedatabase;
    // private TextView tests;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //tests=(TextView)findViewById(R.id.textView2);
        pname=findViewById(R.id.getName);
        paddress=findViewById(R.id.getAddr);
        pmail=findViewById(R.id.getMail);
        ppass=findViewById(R.id.getPass);
        ProfileUpdate=findViewById(R.id.saveData);
        pname.setText("ashish", TextView.BufferType.EDITABLE);

        firebaseauth=FirebaseAuth.getInstance();
        //final String email=((FirebaseUser) firebaseauth.getCurrentUser()).getEmail().toString();
        final String email="rishav.saxena@gmail.com";
        firebasedatabase=FirebaseDatabase.getInstance();
        DatabaseReference databaseReference=firebasedatabase.getReference();
        databaseReference.child("User").addChildEventListener(new ChildEventListener(){
            @Override
            public void onChildAdded(@NonNull com.google.firebase.database.DataSnapshot dataSnapshot, @Nullable String s) {
                UserP up=dataSnapshot.getValue(UserP.class);
                Log.d("Shukla", "onChildAdded: "+up.getEmail());
                if(up.getEmail().equals(email))
                {
                    // tests.setText(up.getName(),TextView.BufferType.EDITABLE);
                    pname.setText("2 X "+"Khus Khus  "+" = "+"98.0");
                    pmail.setText("1 X "+"Liquid Detergent  "+" = "+"273.0");
                    paddress.setText("2 X "+"Moong Dal   "+" = "+"236.0");
                    ppass.setText("1 X "+"Channa Dal   "+"  =  "+"55.0");


                }

            }

            @Override
            public void onChildChanged(@NonNull com.google.firebase.database.DataSnapshot dataSnapshot, @Nullable String s) {

            }

            @Override
            public void onChildRemoved(@NonNull com.google.firebase.database.DataSnapshot dataSnapshot) {

            }

            @Override
            public void onChildMoved(@NonNull com.google.firebase.database.DataSnapshot dataSnapshot, @Nullable String s) {

            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }

        });




    }
    public void onClickBtn(View view)
    {
//        Button v=findViewById(R.id.saveData);
        Toast.makeText(this, "Data Saved Successfully", Toast.LENGTH_LONG).show();
    }


}
