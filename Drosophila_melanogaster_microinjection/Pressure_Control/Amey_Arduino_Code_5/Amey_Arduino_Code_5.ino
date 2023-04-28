/*
  Blink

  Turns an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
  the correct LED pin independent of which board is used.
  If you want to know what pin the on-board LED is connected to on your Arduino
  model, check the Technical Specs of your board at:
  https://www.arduino.cc/en/Main/Products

  modified 8 May 2014
  by Scott Fitzgerald
  modified 2 Sep 2016
  by Arturo Guadalupi
  modified 8 Sep 2016
  by Colby Newman

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/Blink
*/

String readString;
String data, Light, Pressure;
int L, l, P, p, PSI;
int t;
int flag;
int input;
int L_pin = 7;
int P_pin = 11;

void setup() 
{
  // initialize digital pin LED_BUILTIN as an output.
  Serial.begin(9600);
  pinMode(P_pin, OUTPUT);
  pinMode(L_pin, OUTPUT);
}

// the loop function runs over and over again forever
void loop() 
{
  while (!Serial.available()) 
  {
    if (flag == 1)
    {
      digitalWrite(L_pin, HIGH);
    }
    if (flag == 0)
    {
      digitalWrite(L_pin, LOW);
    }
    delay(30);
    /*
    Serial.print("Flag = ");
    Serial.println(flag);
    */
  }
  
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
    P = readString.indexOf("P");
    p = readString.indexOf("p");
    Serial.print("P val:");
    Serial.print(p);
    if (L >= 0)
    {
      Light = readString.substring(L+1,l);
      t = Light.toInt();
      Serial.print(t);
      if (t == 1)
      {
        digitalWrite(L_pin, HIGH);
        flag = 1;
        delay(100);
      }
      if (t == 0)
      {
        digitalWrite(L_pin, LOW);
        flag = 0;
        delay(100);
      }
      Serial.print("T value:");
      Serial.println(t);
    }
    if (p==3 || p==2)
    {
      Pressure = readString.substring(P+1,p);
      PSI = Pressure.toInt();
      Serial.print("Pressure:");
      Serial.print(Pressure);
      Serial.print("PSI:");
      Serial.print(PSI);
      analogWrite(P_pin, PSI);
      delay(500);
      if (PSI == 0)
      {
        flag = 0;
      }
      else 
      {
        flag = 1;
      }
    }
    if (p==-1 || p==0)
    {
      Serial.print("x");
    }
    readString = "";
    /*
    input = readString.toInt();
    Serial.println(input);
    if (input >= 0)
    {
      analogWrite(11, input);
      delay(500);
    }
    readString = "";
    */
  }
}
