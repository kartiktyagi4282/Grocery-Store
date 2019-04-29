package com.example.android.minorsem5;


import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.ChildEventListener;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;


/**
 * A simple {@link Fragment} subclass.
 */
public class AccountDetailsFragment extends Fragment {

    private EditText pname,paddress,pmail,ppass;
    private Button ProfileUpdate;
    private FirebaseAuth firebaseauth;
    private FirebaseDatabase firebasedatabase;

    public AccountDetailsFragment() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View rootView = inflater.inflate(R.layout.fragment_account_details, container, false);

        pname=rootView.findViewById(R.id.getName);
        paddress=rootView.findViewById(R.id.getAddr);
        pmail=rootView.findViewById(R.id.getMail);
        ppass=rootView.findViewById(R.id.getPass);
        ProfileUpdate=rootView.findViewById(R.id.saveData);

        ProfileUpdate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(getActivity(), "Data Saved Successfully", Toast.LENGTH_LONG).show();
            }
        });

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
                    pname.setText(up.getName(),TextView.BufferType.EDITABLE);
                    pmail.setText(up.getEmail(),TextView.BufferType.EDITABLE);
                    paddress.setText(up.getAddress(),TextView.BufferType.EDITABLE);
                    ppass.setText(up.getPassword(),TextView.BufferType.EDITABLE);


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



        return rootView;


    }
//    public void onClickBtn(View view)
//    {
////        Button v=findViewById(R.id.saveData);
//        Toast.makeText(getActivity(), "Data Saved Successfully", Toast.LENGTH_LONG).show();
//    }



    }

