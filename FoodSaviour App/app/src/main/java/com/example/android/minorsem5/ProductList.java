package com.example.android.minorsem5;

import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.Toolbar;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;

public class ProductList extends AppCompatActivity {

    private RecyclerView recyclerview;
    private ArrayList<ProductItem> listItems;
    private DatabaseReference mDataBase;
    private Toolbar mtoolbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_product_list);

        mtoolbar = (Toolbar) findViewById(R.id.product_list_app_bar);
        setSupportActionBar(mtoolbar);
        getSupportActionBar().setTitle("Food Saviour");

        recyclerview = findViewById(R.id.recView2);
        recyclerview.setLayoutManager(new LinearLayoutManager(ProductList.this));
        recyclerview.setHasFixedSize(true);

        listItems = new ArrayList<ProductItem>();

        mDataBase= FirebaseDatabase.getInstance().getReference().child("Products");
        mDataBase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                for(DataSnapshot dataSnapshot1 : dataSnapshot.getChildren()) {
                    ProductItem productItem = dataSnapshot1.getValue(ProductItem.class);
                    listItems.add(productItem);
                }
                //Toast.makeText(MainActivity.this, listItems.get(0).getsName(), Toast.LENGTH_SHORT).show();
               ProductItemAdapter adapter = new ProductItemAdapter(listItems, ProductList.this);
                recyclerview.setAdapter(adapter);
            }
            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });
    }
}
