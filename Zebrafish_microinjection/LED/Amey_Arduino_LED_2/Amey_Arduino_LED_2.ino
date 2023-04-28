#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#define L_pin        6
#define NUMPIXELS  14

String readString;
String Light;
int L, l;
int t;

Adafruit_NeoPixel pixels(NUMPIXELS, L_pin, NEO_GRB + NEO_KHZ800);
#define DELAYVAL 200

void setup() 
{
#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
#endif

  pixels.begin();
  Serial.begin(9600);
  pinMode(L_pin, OUTPUT);
}

void loop() 
{
  while (Serial.available() > 0)
  {
    char c = Serial.read();
    readString += c;
  }
  if (readString.length() > 0)
  {
    Serial.print("Arduino Received:");
    Serial.println(readString);
    L = readString.indexOf("L");
    l = readString.indexOf("l");
    if (L >= 0)
    {
      Light = readString.substring(L+1,l);
      t = Light.toInt();
      Serial.print(t);
      if (t == 1)
      {
        for(int i=0; i<NUMPIXELS; i++) 
        {
          pixels.setPixelColor(i, pixels.Color(255, 255, 255));
          pixels.show();
          delay(DELAYVAL);
        }
        delay(100);
      }
      if (t == 0)
      {
        for(int i=0; i<NUMPIXELS; i++) 
        {
          pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          pixels.show();
          delay(DELAYVAL);
        }
        delay(100);
      }
      Serial.print("T value:");
      Serial.println(t);
    }
  }
  //pixels.clear();
}
