package com.example.myapplication

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.RadioButton
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //將變數與 XML 元件綁定
        val tv_text = findViewById<TextView>(R.id.tv_text)
        val ed_name = findViewById<EditText>(R.id.ed_name)
        val btn_scissor = findViewById<RadioButton>(R.id.btn_scissor)
        val btn_stone = findViewById<RadioButton>(R.id.btn_stone)
        val btn_paper = findViewById<RadioButton>(R.id.btn_paper)
        val btn_mora = findViewById<Button>(R.id.btn_mora)
        val tv_name = findViewById<TextView>(R.id.tv_name)
        val tv_winner = findViewById<TextView>(R.id.tv_winner)
        val tv_mmora = findViewById<TextView>(R.id.tv_mmora)
        val tv_cmora = findViewById<TextView>(R.id.tv_cmora)

        //將監聽器放在猜拳按鈕上
        btn_mora.setOnClickListener {
            //判斷 EditText 的字數是否小於一，若成立則無法進行猜拳
            if (ed_name.length() < 1) {
                tv_text.setText("至少輸入一個字元")

                return@setOnClickListener
            }
            //取出 EditText 文字作為玩家姓名並用變數儲存
            val playerName = ed_name.text
            //亂數產生介於 0~1 之間不含 1 的小數，乘 3 變成 0~2 作為電腦的出拳
            val comMora = (Math.random() * 3).toInt()
            //將玩家出拳結果對應成字串並用變數儲存
            val playerMoraText = when {
                btn_scissor.isChecked -> "剪刀"
                btn_stone.isChecked -> "石頭"
                else -> "布"
            }
            //將電腦出拳結果對應成字串並用變數儲存
            val comMoraText = when(comMora) {
                0 -> "剪刀"
                1 -> "石頭"
                else -> "布"
            }
            //顯示玩家姓名與雙方出拳結果
            tv_name.text = "名字\n$playerName"
            tv_mmora.text = "我方出拳\n$playerMoraText"
            tv_cmora.text = "電腦出拳\n$comMoraText"
            //用三個判斷式決定勝負並顯示猜拳結果
            when {
                btn_scissor.isChecked && comMora == 2 ||
                        btn_stone.isChecked && comMora == 0 ||
                        btn_paper.isChecked && comMora == 1 -> {
                    tv_winner.setText("勝利者\n$playerName")
                    tv_text.setText("恭喜你獲勝了！！！")
                }
                btn_scissor.isChecked && comMora == 1 ||
                        btn_stone.isChecked && comMora == 2 ||
                        btn_paper.isChecked && comMora == 0 -> {
                    tv_winner.setText("勝利者\n 電腦")
                    tv_text.setText("可惜，電腦獲勝了！")
                }
                else -> {
                    tv_winner.setText("勝利者\n 平手")
                    tv_text.setText("平局，請再試一次！")
                }
            }
        }
    }
}