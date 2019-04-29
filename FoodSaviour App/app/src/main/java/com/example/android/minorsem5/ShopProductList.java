package com.example.android.minorsem5;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;
import com.google.firebase.ml.vision.text.RecognizedLanguage;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ShopProductList extends AppCompatActivity {

    private RecyclerView recyclerview;
    private ArrayList<Shop_ProductItem> listItems;
    private DatabaseReference mDataBase;
    private Toolbar mtoolbar;
    private Button check_out;
    private Button take_photo;
    public static ArrayList<String> items;


    private final int TEXT_RECO_REQ_CODE=100;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_shop_product_list);
        check_out=(Button)findViewById(R.id.btn_check_out);
        take_photo=(Button)findViewById(R.id.btn_photo);

        // set app bar with title
        mtoolbar = (Toolbar) findViewById(R.id.shop_product_list_app_bar);
        setSupportActionBar(mtoolbar);
        items=new ArrayList<>();
        getSupportActionBar().setTitle("Products");

        recyclerview = findViewById(R.id.recView3);
        recyclerview.setLayoutManager(new LinearLayoutManager(ShopProductList.this));
        recyclerview.setHasFixedSize(true);

        listItems = new ArrayList<Shop_ProductItem>();

        mDataBase= FirebaseDatabase.getInstance().getReference().child("Products");
        mDataBase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                for(DataSnapshot dataSnapshot1 : dataSnapshot.getChildren()) {
                    Shop_ProductItem shopItem = dataSnapshot1.getValue(Shop_ProductItem.class);
                    listItems.add(shopItem);
                }
                //Toast.makeText(MainActivity.this, listItems.get(0).getsName(), Toast.LENGTH_SHORT).show();
                ShopProductItemAdapter adapter = new ShopProductItemAdapter(listItems, ShopProductList.this);
                recyclerview.setAdapter(adapter);
            }
            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });
        check_out.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                for (Iterator<Map.Entry<String,String>> it = ShopProductItemAdapter.mp_qty.entrySet().iterator(); it.hasNext();) {
                    Map.Entry<String,String> e = it.next();
                    if ("0".equals(e.getValue())) {
                        it.remove();
                    }
                }
                Log.d("Quantity", "onClick: "+ShopProductItemAdapter.mp_qty);
                Intent startIntent = new Intent(ShopProductList.this,YourCart.class);
                startIntent.putExtra("Quantity",ShopProductItemAdapter.mp_qty);
                startIntent.putExtra("Price",ShopProductItemAdapter.mp_price);
                startActivity(startIntent);
           }
        });
        take_photo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ShopProductItemAdapter.mp_qty.clear();
                ShopProductItemAdapter.mp_price.clear();
                textReco(view);
            }

        });
    }


    public void textReco(View view) {
        Intent intent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent,TEXT_RECO_REQ_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==TEXT_RECO_REQ_CODE){
            if(resultCode==RESULT_OK){
                Bitmap photo=(Bitmap)data.getExtras().get("data");
                textRecognition(photo);
            }else if(resultCode==RESULT_CANCELED){
                Toast.makeText(this,"Operation Cancelled By User",Toast.LENGTH_SHORT).show();
            }
            else {
                Toast.makeText(this, "Failed to Capture Image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void textRecognition(Bitmap bitmap) {
        final Random rand = new Random();
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(bitmap);

        FirebaseVisionTextRecognizer textRecognizer = FirebaseVision.getInstance().getOnDeviceTextRecognizer();
        textRecognizer.processImage(image).addOnSuccessListener(new OnSuccessListener<FirebaseVisionText>() {
            @Override
            public void onSuccess(FirebaseVisionText result) {
                String recognisedtext ="";
                String resultText = result.getText();
                for (FirebaseVisionText.TextBlock block: result.getTextBlocks()) {
                    String blockText = block.getText();
                    Float blockConfidence = block.getConfidence();
                    List<RecognizedLanguage> blockLanguages = block.getRecognizedLanguages();
                    Point[] blockCornerPoints = block.getCornerPoints();
                    Rect blockFrame = block.getBoundingBox();
                            /*Toast.makeText(MainActivity.this,"Recognised Text is => "+blockText,Toast.LENGTH_SHORT).show();
                            System.out.println("recognised text is "+blockText);*/

                   /* AlertDialog.Builder builder = new AlertDialog.Builder(ShopProductList.this);
                    builder.setTitle("Text Recognition !");
                    Log.d("test", "onSuccess: "+blockText);
                    recognisedtext += blockText;
                    builder.setMessage("Recognised Text is "+blockText);
                    builder.setCancelable(true);
                    builder.setNeutralButton("Ok", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            Toast.makeText(getApplicationContext(), "Neutral button clicked", Toast.LENGTH_SHORT).show();
                        }
                    });
                    builder.show();*/
                    recognisedtext += blockText;
                    Log.d("test", "onSuccess: "+blockText);
                    for (FirebaseVisionText.Line line: block.getLines()) {
                        String lineText = line.getText();
                        Float lineConfidence = line.getConfidence();
                        List<RecognizedLanguage> lineLanguages = line.getRecognizedLanguages();
                        Point[] lineCornerPoints = line.getCornerPoints();
                        Rect lineFrame = line.getBoundingBox();
                        items.add(lineText);
                        int n = rand.nextInt(200) + 1;
                        ShopProductItemAdapter.mp_qty.put(lineText,"1");
                        ShopProductItemAdapter.mp_price.put(lineText,""+n);
                        Log.d("Items", "onSuccess: "+items);

                        Log.d("line", "reco word: "+items);
                        for (FirebaseVisionText.Element element: line.getElements()) {
                            String elementText = element.getText();

                            Log.d("word", "reco word: "+elementText);
                            Float elementConfidence = element.getConfidence();
                            List<RecognizedLanguage> elementLanguages = element.getRecognizedLanguages();
                            Point[] elementCornerPoints = element.getCornerPoints();
                            Rect elementFrame = element.getBoundingBox();
                        }
                    }
                }
                Log.d("reco", "recog text: "+ recognisedtext);
            }
        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                // ...
                                Toast.makeText(ShopProductList.this,"Failed to Recognise Text from Image !!!",Toast.LENGTH_SHORT).show();
                            }
                        });

    }
}
