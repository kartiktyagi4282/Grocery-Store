package com.example.android.minorsem5;

import android.content.Context;
import android.graphics.Paint;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.squareup.picasso.Picasso;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class ShopProductItemAdapter extends RecyclerView.Adapter<ShopProductItemAdapter.ViewHolder>{
    private List<Shop_ProductItem> listItems;
    private Context context;
    public static HashMap<String,String> mp_qty = new HashMap<>();
    public static HashMap<String,String>mp_price = new HashMap<>();


    public ShopProductItemAdapter(ArrayList<Shop_ProductItem> listItems, Context context) {
        this.listItems = listItems;
        this.context = context;
    }

    @NonNull
    @Override
    public ShopProductItemAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int i) {
        View v= LayoutInflater.from(parent.getContext()).inflate(R.layout.activity_shop__product_item,parent,false);
        return new ViewHolder(v);
    }

   /* @Override
    public void onBindViewHolder(@NonNull ViewHolder viewHolder, int i) {

    }*/

    @Override
    public void onBindViewHolder(@NonNull final ShopProductItemAdapter.ViewHolder viewHolder, int i) {
        /*viewHolder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(viewHolder.itemView.getContext(),ProductList.class);
                viewHolder.itemView.getContext().startActivity(intent);
            }
        });*/
        Shop_ProductItem listItem=listItems.get(i);
        viewHolder.pName.setText(listItem.getProduct_name());
        Log.d("Items in Adapter", ""+ShopProductList.items);
//        if(ShopProductList.items.contains(viewHolder.pName.getText().toString()))
//        {
//            int qty=Integer.parseInt(viewHolder.QtyText.getText().toString());
//            Log.d("shreeshree1008", "onBindViewHolder: "+qty);
//            viewHolder.QtyText.setText(String.valueOf(qty));
//            mp_qty.put(viewHolder.pName.getText().toString() , "1");
//            double netTotal = Double.parseDouble(viewHolder.pDiscountedPrice.getText().toString());
//            double netQuantity = Integer.parseInt(viewHolder.QtyText.getText().toString());
//            netTotal = netTotal * netQuantity;
//            mp_price.put(viewHolder.pName.getText().toString() , String.valueOf(netTotal) );
//        }
        /*if(viewHolder.pName.getText().toString().equals("Khus Khus") || viewHolder.pName.getText().toString().equals("Channa Dal"))
        {
            int qty=Integer.parseInt(viewHolder.QtyText.getText().toString());
            qty=2;

            viewHolder.QtyText.setText(String.valueOf(qty));
            mp_qty.put(viewHolder.pName.getText().toString() , viewHolder.QtyText.getText().toString());
            double netTotal = Double.parseDouble(viewHolder.pDiscountedPrice.getText().toString());


            double netQuantity = Integer.parseInt(viewHolder.QtyText.getText().toString());
            netTotal = netTotal * netQuantity;
            mp_price.put(viewHolder.pName.getText().toString() , String.valueOf(netTotal) );
        }
        else if(viewHolder.pName.getText().toString().equals("Liquid Detergent") || viewHolder.pName.getText().toString().equals("Moong Dal")){
            int qty=Integer.parseInt(viewHolder.QtyText.getText().toString());
            qty=1;
            viewHolder.QtyText.setText(String.valueOf(qty));
            mp_qty.put(viewHolder.pName.getText().toString() , viewHolder.QtyText.getText().toString());
            double netTotal = Double.parseDouble(viewHolder.pDiscountedPrice.getText().toString());


            double netQuantity = Integer.parseInt(viewHolder.QtyText.getText().toString());
            netTotal = netTotal * netQuantity;
            mp_price.put(viewHolder.pName.getText().toString() , String.valueOf(netTotal) );
        }*/
        viewHolder.pDiscountedPrice.setText(""+(listItem.getProduct_discounted_price()));
        viewHolder.pExp.setText(listItem.getExpiry_date());
        viewHolder.pPrice.setText("Rs."+listItem.getProduct_price());
        viewHolder.pPrice.setPaintFlags(Paint.STRIKE_THRU_TEXT_FLAG);
        viewHolder.btnPlus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int qty=Integer.parseInt(viewHolder.QtyText.getText().toString());
                qty++;
                viewHolder.QtyText.setText(String.valueOf(qty));
                mp_qty.put(viewHolder.pName.getText().toString() , viewHolder.QtyText.getText().toString());
                double netTotal = Double.parseDouble(viewHolder.pDiscountedPrice.getText().toString());
                double netQuantity = Integer.parseInt(viewHolder.QtyText.getText().toString());
                netTotal = netTotal * netQuantity;
                mp_price.put(viewHolder.pName.getText().toString() , String.valueOf(netTotal) );
            }
        });
        viewHolder.btnminus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int qty=Integer.parseInt(viewHolder.QtyText.getText().toString());
                if(qty>0)
                    qty--;
                viewHolder.QtyText.setText(String.valueOf(qty));
                mp_qty.put(viewHolder.pName.getText().toString() , viewHolder.QtyText.getText().toString());
                double netTotal = Double.parseDouble(viewHolder.pDiscountedPrice.getText().toString());
                double netQuantity = Double.parseDouble(viewHolder.QtyText.getText().toString());
                netTotal = netTotal * netQuantity;
                mp_price.put(viewHolder.pName.getText().toString() , String.valueOf(netTotal) );
            }
        });

        //Glide.with(viewHolder.itemView).load(listItem.getStores_Image()).thumbnail(Glide.with(viewHolder.itemView).load(R.drawable.ic_launcher_background)).into(viewHolder.sImage);
        Picasso.get()
                .load(listItem.getProduct_image())
                .placeholder(R.drawable.food_product)
                .error(R.drawable.food_product)
               .into(viewHolder.pImage);
    }

    @Override
    public int getItemCount() {
        return listItems.size();
    }
    public class ViewHolder extends RecyclerView.ViewHolder{
        public TextView pName;
        //public TextView pQuantity;
        //public TextView pMfg;
        public TextView pDiscountedPrice;
        public TextView pExp;
        public TextView pPrice;
        public ImageView pImage;
        public TextView QtyText;
        public Button btnPlus;
        public Button btnminus;


        //public ImageView sImage;
        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            pName=(TextView)itemView.findViewById(R.id.spName);
            pDiscountedPrice=(TextView)itemView.findViewById(R.id.spDisPrice);
            pExp=(TextView)itemView.findViewById(R.id.spExp);
            pPrice=(TextView)itemView.findViewById(R.id.spPrice);
            pImage=(ImageView)itemView.findViewById(R.id.iv2);
            btnPlus =(Button)itemView.findViewById(R.id.btnplus);
            btnminus =(Button)itemView.findViewById(R.id.btnminus);
            QtyText = (TextView)itemView.findViewById(R.id.QtyText);
            Log.d("aman6", "ViewHolder: Constructor");

        }
    }
}
