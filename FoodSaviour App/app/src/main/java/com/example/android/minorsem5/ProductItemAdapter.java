package com.example.android.minorsem5;

import android.content.Context;
import android.graphics.Paint;
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.squareup.picasso.Picasso;

import java.util.ArrayList;
import java.util.List;

public class ProductItemAdapter extends RecyclerView.Adapter<ProductItemAdapter.ViewHolder>{
    private List<ProductItem> listItems;
    private Context context;

    public ProductItemAdapter(ArrayList<ProductItem> listItems, Context context) {
        this.listItems = listItems;
        this.context = context;
    }


    @NonNull
    @Override
    public ProductItemAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int i) {
        View v= LayoutInflater.from(parent.getContext()).inflate(R.layout.activity_product_item,parent,false);
        return new ViewHolder(v);
    }

   /* @Override
    public void onBindViewHolder(@NonNull ViewHolder viewHolder, int i) {

    }*/

    @Override
    public void onBindViewHolder(@NonNull final ProductItemAdapter.ViewHolder viewHolder, int i) {
        /*viewHolder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(viewHolder.itemView.getContext(),ProductList.class);
                viewHolder.itemView.getContext().startActivity(intent);
            }
        });*/
        ProductItem listItem=listItems.get(i);
        viewHolder.pName.setText(listItem.getProduct_name());
        //viewHolder.pQuantity.setText(listItem.getProduct_Quantity() + "");
        //viewHolder.pMfg.setText(listItem.getProduct_Mfg());
        viewHolder.pDiscountedPrice.setText("Rs "+listItem.getProduct_discounted_price());

        viewHolder.pExp.setText(listItem.getExpiry_date());
        viewHolder.pPrice.setText("Rs."+listItem.getProduct_price());
        viewHolder.pPrice.setPaintFlags(Paint.STRIKE_THRU_TEXT_FLAG);
        //Glide.with(viewHolder.itemView).load(listItem.getStores_Image()).thumbnail(Glide.with(viewHolder.itemView).load(R.drawable.ic_launcher_background)).into(viewHolder.sImage);
        Picasso.get()
                .load(listItem.getProduct_image())
                //.placeholder()
                //.error() n
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
        //public ImageView sImage;
        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            pName=(TextView)itemView.findViewById(R.id.pName);
            pDiscountedPrice=(TextView)itemView.findViewById(R.id.pDisPrice);
            pExp=(TextView)itemView.findViewById(R.id.pExp);
            pPrice=(TextView)itemView.findViewById(R.id.pPrice);
            pImage=(ImageView)itemView.findViewById(R.id.product_item_image);
        }
    }
}
