package com.example.android.minorsem5;

import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;

public class CartItemAdapter extends RecyclerView.Adapter<CartItemAdapter.CartViewHolder> {

    ArrayList<CartItem> listItem;

    public CartItemAdapter(ArrayList<CartItem> listItem) {
        this.listItem = listItem;
    }


    @NonNull
    @Override
    public CartViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int i) {
        View v= LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.activity_cart_item,viewGroup,false);
        return new CartViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull CartViewHolder cartViewHolder, int i) {
        CartItem list=listItem.get(i);
        cartViewHolder.tv.setText(list.getItem_details());
    }

    @Override
    public int getItemCount() {
        return listItem.size();
    }

    public class CartViewHolder extends RecyclerView.ViewHolder {
        TextView tv;
        public CartViewHolder(@NonNull View itemView) {
            super(itemView);
            tv=(TextView) itemView.findViewById(R.id.yourcart_textview);
        }
    }
}
