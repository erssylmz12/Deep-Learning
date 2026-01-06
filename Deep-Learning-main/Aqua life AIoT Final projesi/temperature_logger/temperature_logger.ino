#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_BUS 8

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
unsigned long startMillis;

void setup() {
  Serial.begin(9600);
  sensors.begin();
  startMillis = millis();
  Serial.println("timestamp,temperature_C");
}

void loop() {
  if (millis() - startMillis >= 6000) {
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);
    unsigned long ts = millis() / 1000; 
    Serial.print(ts);
    Serial.print(",");
    Serial.println(tempC, 1);
    startMillis += 1200;
  }
}