package com.example.android.minorsem5;

import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ChatViewHolder> {


    ArrayList<ChatMessage> chatlist;
    View view;
    public ChatAdapter(ArrayList<ChatMessage> chatList) {

        chatlist=chatList;
    }
    @NonNull
    @Override
    public ChatViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int i) {
        view= LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.msglist,viewGroup,false);
        return new ChatViewHolder(view);
    }
    @Override
    public void onBindViewHolder(@NonNull ChatViewHolder holder, int position) {
        if (chatlist.get(position).getMsgUser().equals("user")) {
            //Log.d("Test",chatlist.get(position).getMsgText()+"From User");
            holder.rightText.setText(chatlist.get(position).getMsgText());
            holder.rightText.setVisibility(View.VISIBLE);
            holder.leftText.setVisibility(View.GONE);
            holder.avatar1.setVisibility(View.VISIBLE);
            holder.avatar.setVisibility(View.GONE);
            holder.name.setVisibility(View.GONE);
            holder.name1.setVisibility(View.VISIBLE);
        }
        else {
            //Log.d("Test",chatlist.get(position).getMsgText()+"From Server");
            holder.leftText.setText(chatlist.get(position).getMsgText());
            holder.rightText.setVisibility(View.GONE);
            holder.leftText.setVisibility(View.VISIBLE);
            holder.avatar1.setVisibility(View.GONE);
            holder.avatar.setVisibility(View.VISIBLE);
            holder.name.setVisibility(View.VISIBLE);
            holder.name1.setVisibility(View.GONE);
        }
    }
    @Override
    public int getItemCount() {
        return chatlist.size();
    }

    public class ChatViewHolder extends RecyclerView.ViewHolder {
        TextView leftText,rightText,name,name1;
        View avatar,avatar1;
        public ChatViewHolder(@NonNull View itemView) {
            super(itemView);
            leftText=itemView.findViewById(R.id.leftText);
            rightText=itemView.findViewById(R.id.rightText);
            name=itemView.findViewById(R.id.ChatBot_name);
            name1=itemView.findViewById(R.id.ChatUser_name);
            avatar=itemView.findViewById(R.id.avatar);
            avatar1=itemView.findViewById(R.id.avatar1);
        }
    }
}