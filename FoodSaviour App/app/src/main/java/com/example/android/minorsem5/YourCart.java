package com.example.android.minorsem5;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class YourCart extends AppCompatActivity {

    private RecyclerView recyclerView;
    private Toolbar mtoolbar;
    private Button place_order;
    private TextView shop_name;
    private TextView grand_total;
    private ArrayList<CartItem> listItems;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_your_cart);
        listItems=new ArrayList<CartItem>();
        HashMap<String,String> mp_qty=(HashMap<String, String>) (getIntent().getSerializableExtra("Quantity"));
        HashMap<String,String> mp_price=(HashMap<String, String>)(getIntent().getSerializableExtra("Price"));
        // set app bar with title
        Log.d("pName", "quantity:"+mp_qty.size());
        Log.d("pName","price:"+mp_price.size());
        mtoolbar = (Toolbar) findViewById(R.id.yourCart_app_bar);
        setSupportActionBar(mtoolbar);
        getSupportActionBar().setTitle("Your Cart");
//        listItems.add(new CartItem("Rishav"));
//        listItems.add(new CartItem("Aman"));
//        listItems.add(new CartItem("Shubham"));
        recyclerView = findViewById(R.id.rView2);
        recyclerView.setHasFixedSize(true);
        recyclerView.setLayoutManager(new LinearLayoutManager(YourCart.this));
        place_order = findViewById(R.id.btnPlace);
        shop_name = findViewById(R.id.shopName);
        grand_total = findViewById(R.id.grand_total);
        CartItemAdapter adapter=new CartItemAdapter(listItems);
        double bill=0.0;

        shop_name.setText("Your Selected Items");
        for(Map.Entry<String,String> entry : mp_qty.entrySet()){
            CartItem tv=new CartItem(""+entry.getValue()+"  X  "+ entry.getKey()+ "    "+mp_price.get(entry.getKey()));
            Log.d("pName", entry.getKey()+" "+entry.getValue());
            listItems.add(tv);
            adapter.notifyDataSetChanged();
        }

        recyclerView.setAdapter(adapter);
        for(Map.Entry<String,String> entry : mp_price.entrySet()){
            bill=bill+Double.parseDouble(entry.getValue());
        }
        grand_total.setText("Grand Total"+"    "+bill);
        place_order.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // last intent here
            }
        });
    }
}
