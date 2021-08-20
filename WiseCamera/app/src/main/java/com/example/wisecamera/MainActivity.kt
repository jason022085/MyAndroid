package com.example.wisecamera

import android.content.Context
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.Intent.ACTION_PICK
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.core.view.drawToBitmap
import com.example.wisecamera.ml.Eff0819CarMlMetadata
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import org.tensorflow.lite.support.image.TensorImage
import java.io.IOException
import android.graphics.drawable.BitmapDrawable




class MainActivity : AppCompatActivity() {
    private var angle = 0f
    //取得返回的影像資料
    override fun onActivityResult(requestCode: Int,
                                  resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        //識別返回對象及執行結果
        if (requestCode == 0 && resultCode == RESULT_OK) {
            val image = data?.extras?.get("data") ?: return //取得資料
            val bitmap = image as Bitmap //將資料轉換成 Bitmap
            val imageView = findViewById<ImageView>(R.id.imageView)
            imageView.setImageBitmap(bitmap) //使用 Bitmap 設定圖像
            recognizeImage(bitmap) //使用 Bitmap 進行辨識

        }
        if (requestCode == 1 && resultCode == RESULT_OK) {
            val uri = data!!.data
            val imageView = findViewById<ImageView>(R.id.imageView)
            imageView.setImageURI(uri)
            val drawable = imageView.drawable as BitmapDrawable //從imageView取得資料，轉換成Bitmap
            val bitmap = drawable.bitmap
            recognizeImage(bitmap) //使用 Bitmap 進行辨識
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
            findViewById<Button>(R.id.btn_photo).setOnClickListener {
                //建立一個要進行影像獲取的 Intent 物件
                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                //用 try-catch 避免例外產生，若產生則顯示 Toast
                try {
                    startActivityForResult(intent, 0) //發送 Intent
                } catch (e: ActivityNotFoundException) {
                    Toast.makeText(
                        this,
                        "此裝置無相機應用程式", Toast.LENGTH_SHORT
                    ).show()
                }
            }

            findViewById<Button>(R.id.btn_upload).setOnClickListener {
                //建立一個要進行影像獲取的 Intent 物件
                val intent =
                    Intent(Intent.ACTION_GET_CONTENT).setType("image/*")
                //用 try-catch 避免例外產生，若產生則顯示 Toast
                try {
                    startActivityForResult(intent, 1) //發送 Intent
                } catch (e: ActivityNotFoundException) {
                    Toast.makeText(
                        this,
                        "請選擇一張照片", Toast.LENGTH_SHORT
                    ).show()
                }
            }

            findViewById<Button>(R.id.btn_rotate).setOnClickListener {
                val imageView = findViewById<ImageView>(R.id.imageView)
                angle += 90f //原本角度再加上 90 度
                imageView.rotation = angle //使 ImageView 旋轉
                val bitmap = imageView.drawToBitmap() //取得 Bitmap
                recognizeImage(bitmap) //使用 Bitmap 進行辨識
            }
        }


    //辨識圖像
    private fun recognizeImage(bitmap: Bitmap) {
        try {
            //1.For my custom model
            val model = Eff0819CarMlMetadata.newInstance(this)

            // Creates inputs for reference.
            val tensorImage = TensorImage.fromBitmap(bitmap)

            // Runs model inference and gets result.
            val outputs = model.process(tensorImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // 排序，由高到低
                }

            //取得辨識結果與可信度
            val result = arrayListOf<String>()
            for (output in outputs) {
                val label = output.label
                val score = output.score
                result.add("$label 可信度：$score")
            }

            //將結果顯示於 ListView
            val listView = findViewById<ListView>(R.id.listView)
            listView.adapter = ArrayAdapter(this,
                android.R.layout.simple_list_item_1,
                result
            )

            //2.For TFHub downloaded model
//            //取得自訂的模型
//            val localModel = LocalModel.Builder()
//                .setAbsoluteFilePath("C:\\Users\\USER\\AndroidStudioProjects\\WiseCamera\\app\\src\\main\\ml\\eff_0819_car_ml_metadata.tflite")
//                .build()
//
//            //建立自訂的標籤
//            val customImageLabelerOptions = CustomImageLabelerOptions.Builder(localModel)
//                .build()
//
//            //取得辨識標籤
//            val labeler = ImageLabeling.getClient(customImageLabelerOptions)
//            //val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
//
//            //建立 InputImage 物件
//            val inputImage = InputImage.fromBitmap(bitmap, 0)
//            //匹配辨識標籤與圖像，並建立執行成功與失敗的監聽器
//            labeler.process(inputImage)
//                .addOnSuccessListener { labels ->
//                    //取得辨識結果與可信度
//                    val result = arrayListOf<String>()
//                    for (label in labels) {
//                        val text = label.text
//                        val confidence = label.confidence
//                        result.add("$text, 可信度：$confidence")
//                    }
//                    //將結果顯示於 ListView
//                    val listView = findViewById<ListView>(R.id.listView)
//                    listView.adapter = ArrayAdapter(this,
//                        android.R.layout.simple_list_item_1,
//                        result
//                    )
//
//                }
//                .addOnFailureListener { e ->
//                    Toast.makeText(this,
//                        "發生錯誤", Toast.LENGTH_SHORT).show()
//                }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}