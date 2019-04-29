package com.example.android.minorsem5;


import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.Fragment;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.squareup.picasso.Picasso;

import java.util.ArrayList;
import java.util.List;


/**
 * A simple {@link Fragment} subclass.
 */
public class AllStoreFragment extends Fragment {

    private ArrayList<ListItem> listItems;
    private DatabaseReference mDataBase;
    private RecyclerView recyclerView;

    public AllStoreFragment() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        // Inflate the layout for this fragment
        View rootView = inflater.inflate(R.layout.fragment_all_store, container, false);

        recyclerView = (RecyclerView) rootView.findViewById(R.id.recView);

        recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
        recyclerView.setHasFixedSize(true);

        listItems = new ArrayList<>();

        mDataBase= FirebaseDatabase.getInstance().getReference().child("ID");

        mDataBase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                for(DataSnapshot dataSnapshot1 : dataSnapshot.getChildren()){
                    ListItem listItem = dataSnapshot1.getValue(ListItem.class);
                    listItems.add(listItem);
                }
                //Toast.makeText(MainActivity.this, listItems.get(0).getsName(), Toast.LENGTH_SHORT).show();
                recyclerView.setAdapter(new MyAdapter(listItems, getActivity()));
            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });



        return rootView;
    }

    public class MyAdapter extends RecyclerView.Adapter<MyAdapter.ViewHolder>{
        private List<ListItem> listItems;
        private Context context;

        public MyAdapter(List<ListItem> listItems, Context context) {
            this.listItems = listItems;
            this.context = context;
        }


        @NonNull
        @Override
        public MyAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int i) {
            View v=LayoutInflater.from(parent.getContext()).inflate(R.layout.activity_list_item,parent,false);
            return new ViewHolder(v);
        }

        @Override
        public void onBindViewHolder(@NonNull final MyAdapter.ViewHolder viewHolder, int i) {

            viewHolder.itemView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Intent intent=new Intent(viewHolder.itemView.getContext(),ProductList.class);
                    viewHolder.itemView.getContext().startActivity(intent);
                }
            });

            ListItem listItem=listItems.get(i);
            viewHolder.sName.setText(listItem.getStores_name());
            viewHolder.sRating.setText(" "+listItem.getStores_Rating() + "");
            viewHolder.sAddress.setText(listItem.getStores_Address());
            viewHolder.sContact.setText(listItem.getStores_Contact());
            //Glide.with(viewHolder.itemView).load(listItem.getStores_Image()).thumbnail(Glide.with(viewHolder.itemView).load(R.drawable.ic_launcher_background)).into(viewHolder.sImage);
            Picasso.get()
                    .load(listItem.getStores_Image())
                    .placeholder(R.drawable.store)
                    .error(R.drawable.store)
                    .into(viewHolder.sImage);
        }

        @Override
        public int getItemCount() {
            return listItems.size();
        }
        public class ViewHolder extends RecyclerView.ViewHolder{
            public TextView sName;
            public TextView sRating;
            public TextView sAddress;
            public TextView sContact;
            public ImageView sImage;
            public ViewHolder(@NonNull View itemView) {
                super(itemView);
                sName=(TextView)itemView.findViewById(R.id.sName);
                sRating=(TextView)itemView.findViewById(R.id.sRating);
                sAddress=(TextView)itemView.findViewById(R.id.sAddress);
                sContact=(TextView)itemView.findViewById(R.id.sContact);
                sImage=(ImageView)itemView.findViewById(R.id.iv1);
            }
        }
    }

}
