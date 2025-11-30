#include <Arduino.h>

void setup(){
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop(){
    if(Serial.available() > 0){
        char data = Serial.read();

        if(data == 'O'){
            digitalWrite(LED_BUILTIN, HIGH);
        }
        else if(data == 'C'){
            digitalWrite(LED_BUILTIN, LOW);
        }
    }
}