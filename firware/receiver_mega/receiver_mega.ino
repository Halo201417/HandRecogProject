#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup(){
    Serial.begin(9600);
    
    lcd.init();
    lcd.backlight();

    lcd.setCursor(0,0);
    lcd.print("Translator");
    lcd.setCursor(0,1);
    lcd.print("Waiting...");
    delay(2000);
    lcd.clear();
}

void loop(){
    if(Serial.available() > 0){
        char data = Serial.read();

        lcd.setCursor(0,0);
        lcd.print("Gesture detect:");

        lcd.setCursor(7,1);
        lcd.print(data);
    }
}