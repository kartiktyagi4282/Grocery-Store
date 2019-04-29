package com.example.android.minorsem5;


import android.content.Context;
import android.content.Intent;
import android.location.Location;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.Fragment;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
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
public class RequestsFragment extends Fragment {

    private ArrayList<ListItem> listItems;
    private DatabaseReference mDataBase;
    private DatabaseReference mDataBase_dimension;
    private RecyclerView recyclerView;
    private ArrayList<LatLang>arr_dimension;
   // private String user;
    //private DatabaseReference databaseReference;
    public RequestsFragment() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        // Inflate the layout for this fragment
        View rootView = inflater.inflate(R.layout.fragment_requests, container, false);

        recyclerView = (RecyclerView) rootView.findViewById(R.id.fragment_request_recView);
        recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
        recyclerView.setHasFixedSize(true);

        listItems = new ArrayList<>();
        arr_dimension = new ArrayList<>();

        mDataBase= FirebaseDatabase.getInstance().getReference().child("ID");
        mDataBase_dimension= FirebaseDatabase.getInstance().getReference().child("Dimensions");

        mDataBase_dimension.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                for(DataSnapshot dataSnapshot1 : dataSnapshot.getChildren()){
                    LatLang lat_lang = dataSnapshot1.getValue(LatLang.class);
                    arr_dimension.add(lat_lang);
                    Log.d("mickey", "onDataChange: "+arr_dimension.size());
                }

            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });
        mDataBase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                int cnt=-1;

                for(DataSnapshot dataSnapshot1 : dataSnapshot.getChildren()){
                    ListItem listItem = dataSnapshot1.getValue(ListItem.class);
                    Log.d("mickey2", "onDataChange: "+cnt);

                    // ******************geo location******************
                    cnt++;
                    if(cnt<arr_dimension.size()) {
                        LatLang store_coordinates = arr_dimension.get(cnt);

                        Log.d("mickey3", "onDataChange: "+cnt);

//                    String store_address = listItem.getStores_Address();

//                   LatLng store_coordinates = getLocationFromAddress(getContext(),store_address);
//                   if(store_coordinates!=null) {
//                      //  Log.d("TAG", store_coordinates.toString());
//
//                        LatLng user_coordinates = getLocationFromAddress(getContext(),"Jaypee Institute Of Information Technology");
//                      //  Log.d("TAG", user_coordinates.toString());
//
                        double lat_store = Double.parseDouble(store_coordinates.getLat());
                        double lng_store = Double.parseDouble(store_coordinates.getLong());
                        double lat_user = 28.6295406;
                        double lng_user = 77.37250483714578;

                        Location locationStore = new Location("point A");
                        locationStore.setLatitude(Double.parseDouble(store_coordinates.getLat()));
                        locationStore.setLongitude(Double.parseDouble(store_coordinates.getLong()));

                        Location locationUser = new Location("point B");
                        locationUser.setLatitude(lat_user);
                        locationUser.setLongitude(lng_user);
//
                        double distance = locationUser.distanceTo(locationStore);
                        if (distance / 1000 <= 10) {
                            listItems.add(listItem);
                        }
                    }
                   }
  //              }
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
                    ShopProductItemAdapter.mp_qty.clear();
                    ShopProductItemAdapter.mp_price.clear();
                    Intent intent=new Intent(viewHolder.itemView.getContext(),ShopProductList.class);
                    viewHolder.itemView.getContext().startActivity(intent);

                }
            });

            ListItem listItem=listItems.get(i);
            viewHolder.sName.setText(listItem.getStores_name());
            viewHolder.sRating.setText(listItem.getStores_Rating() + "");
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


    /*public LatLng getLocationFromAddress(Context context, String strAddress) {

        Geocoder coder = new Geocoder(context);
        List<Address> address=null;
        LatLng p1 = null;

        try {
            address = coder.getFromLocationName(strAddress, 1);
            if (address == null) {
                return null;
            }
            Address location = address.get(0);
           // location.getLatitude();
            //location.getLongitude();

            p1 = new LatLng(location.getLatitude(), location.getLongitude() );

        } catch (Exception ex) {

            ex.printStackTrace();
        }

        return p1;
    }
*/
  /*  class LatLang{
       String latitude;
        String longitude;

        public LatLang() {
            latitude="";
            longitude="";

        }

        public String getLatitude() {
            return latitude;
        }

        public void setLatitude(String latitude) {
            this.latitude = latitude;
        }

        public String getLongitude() {
            return longitude;
        }

        public void setLongitude(String longitude) {
            this.longitude = longitude;
        }
    }*/


}

